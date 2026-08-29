"""Thin, retrying, usage-accounting wrapper around the Gemini API.

Everything the pipeline needs from a model goes through `json()`. Keeping it
narrow is what makes the model swappable, and swapping models is a measurement
we intend to run, not a rewrite we intend to avoid.
"""

from __future__ import annotations

import hashlib
import io
import os
import random
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from PIL import Image
from pydantic import BaseModel

from ..config import DEFAULT_MODEL  # noqa: E402  (re-exported for callers)

# gemini-3.5-flash is kept only to replicate earlier recorded runs; it costs
# materially more than 3.7 for this workload.
REPLICATION_MODEL = "gemini-3.5-flash"
# A request with no deadline can block a worker forever; observed in a full
# benchmark pass where six workers wedged on network I/O at 0% CPU.
REQUEST_TIMEOUT_S = 180
_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Retried: transient server and rate-limit conditions.
_RETRY_MARKERS = ("429", "500", "502", "503", "504", "UNAVAILABLE",
                  "RESOURCE_EXHAUSTED", "INTERNAL", "DEADLINE_EXCEEDED",
                  "timeout", "Timeout", "TimeoutError", "ReadTimeout", "ConnectError")


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Read a prompt once per process.

    Prompts were previously re-read on every call, so editing one while an
    evaluation was in flight silently changed the system half way through the
    run and produced a result that measured two different pipelines.
    """
    return (_PROMPT_DIR / name).read_text()


@lru_cache(maxsize=1)
def prompts_fingerprint() -> str:
    """Short hash over all prompt text, stamped into every response so a result
    can be tied back to the prompts that produced it."""
    digest = hashlib.sha256()
    for path in sorted(_PROMPT_DIR.glob("*.md")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


@dataclass
class Usage:
    """Token accounting, so cost per video-hour is measured rather than guessed."""

    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0
    by_stage: dict[str, int] = field(default_factory=dict)
    # Stages fan out across threads, so accounting is guarded.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, stage: str, prompt: int, output: int) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += prompt
            self.output_tokens += output
            self.by_stage[stage] = self.by_stage.get(stage, 0) + 1

    def note_retry(self) -> None:
        with self._lock:
            self.retries += 1

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "retries": self.retries,
            "calls_by_stage": dict(self.by_stage),
        }


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoClip:
    """A video part for the model: encoded bytes plus the frame rate to sample at."""

    data: bytes
    fps: float
    mime_type: str = "video/mp4"


class _Transient(LLMError):
    """Retryable regardless of message; see _RETRY_MARKERS for the rest."""


def png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


class GeminiClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        temperature: float | None = 0.0,
        max_retries: int = 4,
        timeout_s: float = REQUEST_TIMEOUT_S,
        thinking_level: str | None = None,
        thinking_budget: int | None = None,
        media_resolution: str | None = None,
        media_processing: str | None = None,
    ) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY is not set (see .env.example)")
        from google import genai  # imported lazily so video-only use needs no SDK

        from google.genai import types as _types

        self._genai = genai
        self.timeout_s = timeout_s
        self.client = genai.Client(
            api_key=key,
            # google-genai expects milliseconds here.
            http_options=_types.HttpOptions(timeout=int(timeout_s * 1000)),
        )
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.thinking_level = thinking_level
        self.thinking_budget = thinking_budget
        # "low" | "medium" | "high": token budget per frame / image.
        self.media_resolution = media_resolution
        # "static" (fixed-rate frame extraction) | "agentic" (model-driven). Video only.
        self.media_processing = media_processing
        self.usage = Usage()

    def json(
        self,
        stage: str,
        text: str,
        schema: type[BaseModel],
        images: list[Image.Image] | None = None,
        videos: list[VideoClip] | None = None,
    ) -> BaseModel:
        """One structured-output call. Returns a validated pydantic model.

        Images follow the text (the measured contact-sheet layout). Video goes
        before the text, which is Google's documented best practice for long
        media: instructions last, after the data.
        """
        from google.genai import types

        parts: list = []
        for clip in videos or []:
            kw = {}
            if self.media_processing:
                kw["media_processing"] = types.MediaProcessing(self.media_processing.upper())
            parts.append(types.Part(
                inline_data=types.Blob(data=clip.data, mime_type=clip.mime_type),
                video_metadata=types.VideoMetadata(fps=clip.fps),
                **kw,
            ))
        parts.append(types.Part.from_text(text=text))
        for img in images or []:
            parts.append(types.Part.from_bytes(data=png_bytes(img), mime_type="image/png"))

        thinking = None
        if self.thinking_level is not None or self.thinking_budget is not None:
            thinking = types.ThinkingConfig(
                thinking_level=self.thinking_level, thinking_budget=self.thinking_budget,
            )
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            # None leaves the API default (recommended for gemini-3.x, where the
            # parameter is deprecated and values below 1.0 are discouraged).
            temperature=self.temperature,
            thinking_config=thinking,
            media_resolution=(
                types.MediaResolution(f"MEDIA_RESOLUTION_{self.media_resolution.upper()}")
                if self.media_resolution else None
            ),
            http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)),
        )

        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=[types.Content(role="user", parts=parts)],
                    config=config,
                )
                meta = getattr(resp, "usage_metadata", None)
                self.usage.add(
                    stage,
                    getattr(meta, "prompt_token_count", 0) or 0,
                    (getattr(meta, "candidates_token_count", 0) or 0)
                    + (getattr(meta, "thoughts_token_count", 0) or 0),
                )
                parsed = getattr(resp, "parsed", None)
                if parsed is not None:
                    return parsed
                if resp.text:
                    return schema.model_validate_json(resp.text)
                # An empty candidate is intermittent rather than terminal -- the
                # same request succeeds on a retry -- so it is treated as such.
                finish = [str(c.finish_reason) for c in (resp.candidates or [])]
                raise _Transient(f"{stage}: model returned no content (finish={finish})")
            except Exception as exc:  # noqa: BLE001 - retry policy is marker-based
                last = exc
                retryable = isinstance(exc, _Transient) or any(
                    m in str(exc) for m in _RETRY_MARKERS
                )
                if not retryable or attempt == self.max_retries - 1:
                    break
                self.usage.note_retry()
                time.sleep(min(2**attempt + random.random(), 20))
        raise LLMError(f"{stage} failed after {self.max_retries} attempts: {last}") from last
