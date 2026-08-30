"""Hugging Face Space front end: the same `rel.pipeline.annotate` as the CLI and
the Replicate model, behind a Gradio form.

Bring-your-own-key: the caller's Gemini key is passed to the client for that
call only. It is never stored, logged, or put in the process environment (which
would be shared between concurrent requests).
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import traceback

import gradio as gr

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # `rel/` is copied next to this file at publish time
# Running from the repository checkout instead (deploy/hf_space/app.py): use src/.
# On the Space the file sits at /app/app.py, where there is no such parent.
if len(HERE.parents) >= 2 and (HERE.parents[1] / "src" / "rel").is_dir():
    sys.path.insert(0, str(HERE.parents[1] / "src"))

from rel.annotation.llm import LLMError  # noqa: E402
from rel.config import config_for  # noqa: E402
from rel.pipeline import annotate, client_for  # noqa: E402
from rel.schemas import AnnotateRequest, Quality  # noqa: E402
from rel.video.decode import VideoError, probe  # noqa: E402

# Free CPU hardware and a synchronous UI: keep episodes to a few minutes.
MAX_DURATION_S = 300
MAX_BYTES = 200 * 1024 * 1024
EXAMPLES_DIR = HERE / "examples"


def _split(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def run(video, prompt, subtasks, attributes, quality, gemini_api_key):
    # Bring-your-own-key only. There is deliberately no fallback to a server-side
    # key: a public form backed by the operator's key would be an open tap on
    # the operator's Gemini account.
    key = (gemini_api_key or "").strip()
    if not key:
        raise gr.Error("A Gemini API key is required. Get one at https://aistudio.google.com/apikey")
    if not video:
        raise gr.Error("Upload an episode video first.")
    path = video if isinstance(video, str) else getattr(video, "name", str(video))
    if os.path.getsize(path) > MAX_BYTES:
        raise gr.Error(f"Video is larger than {MAX_BYTES // 2**20} MB; cut it down first.")
    try:
        info = probe(path)
    except VideoError as exc:
        raise gr.Error(f"Could not read the video: {exc}") from exc
    if info.duration > MAX_DURATION_S:
        raise gr.Error(f"Episode is {info.duration:.0f}s; this demo caps episodes at "
                       f"{MAX_DURATION_S}s. Run the package locally for longer videos.")

    request = AnnotateRequest(
        video=path, prompt=prompt or "", subtasks=_split(subtasks),
        attributes=_split(attributes), quality=Quality(quality),
    )
    cfg = config_for(request.quality)
    try:
        response = annotate(request, client=client_for(cfg, api_key=key), config=cfg)
    except LLMError as exc:
        msg = str(exc)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            raise gr.Error("Gemini rejected the request (rate limit or exhausted credits on "
                           "the key you supplied). Check the key's project at "
                           "https://aistudio.google.com/") from exc
        if "API key" in msg or "401" in msg or "403" in msg or "PERMISSION_DENIED" in msg:
            raise gr.Error("Gemini rejected the API key.") from exc
        raise gr.Error(f"Model call failed: {msg[:300]}") from exc
    except VideoError as exc:
        raise gr.Error(f"Video processing failed: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - surface, never swallow, in a demo
        traceback.print_exc()
        raise gr.Error(f"Unexpected error: {type(exc).__name__}: {exc}"[:300]) from exc

    rows = [
        [f"{s.start_seconds:.2f}", f"{s.end_seconds:.2f}", s.label, s.result.value,
         ", ".join(s.attributes), s.confidence.value, ", ".join(s.flags), s.description]
        for s in response.segments
    ]
    usage = response.metadata.get("usage", {})
    summary = (
        f"{len(response.segments)} subtasks over {response.duration_seconds:.1f}s · "
        f"{usage.get('calls', 0)} model calls · {usage.get('prompt_tokens', 0):,} in / "
        f"{usage.get('output_tokens', 0):,} out tokens · "
        f"{response.metadata.get('elapsed_seconds', 0):.0f}s"
    )
    warnings = "\n".join(f"• {w}" for w in response.warnings) or "none"
    return rows, summary, warnings, json.loads(response.model_dump_json())


def _examples() -> list[list]:
    ex = []
    a = EXAMPLES_DIR / "droid_blue_block_in_green_bowl.mp4"
    b = EXAMPLES_DIR / "droid_duvet_tip_left.mp4"
    if a.exists():
        ex.append([str(a), "Put the blue block in the green bowl",
                   "Pick Up Block,Place Block In Bowl", "retry,missed_grasp,dropped_object", "balanced"])
        ex.append([str(a), "Put the blue block in the green bowl", "", "", "balanced"])
    if b.exists():
        ex.append([str(b), "Move the bottom right tip of the duvet to the left",
                   "Grasp Duvet,Drag Duvet,Release Duvet", "retry,missed_grasp,dropped_object", "balanced"])
    return ex


DESCRIPTION = """
Turn a robot manipulation video into timestamped subtasks: start, end, a label,
pass/fail, and failure attributes. Send an episode and one sentence describing
the task. Supplying **subtasks** constrains labels to your own vocabulary
(schema mode); leave it empty and the model names the events itself.

This demo calls the Gemini API with **your** key (bring-your-own-key); the key
is used for this call only and never stored. Episodes are capped at 5 minutes
here — run the [package](https://github.com/Manda-Robotics/robot-episode-labeler)
locally for more. Accuracy, limits and data handling are described below the form.
"""

with gr.Blocks(title="Robot Episode Labeler") as demo:
    gr.Markdown("# Robot Episode Labeler")
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column(scale=1):
            video = gr.Video(label="Episode video (mp4 / mov / webm)", sources=["upload"])
            prompt = gr.Textbox(label="Task", placeholder="Put the blue block in the green bowl")
            subtasks = gr.Textbox(label="Subtask vocabulary (optional, comma-separated)",
                                  placeholder="Pick Up Block,Place Block In Bowl")
            attributes = gr.Textbox(label="Attribute rubric (optional, comma-separated)",
                                    placeholder="retry,missed_grasp,dropped_object")
            quality = gr.Radio(["fast", "balanced", "strict"], value="balanced", label="Quality",
                               info="fast = segmentation only · balanced = + subdivision + labels · strict = + refinement")
            key = gr.Textbox(label="Gemini API key", type="password",
                             placeholder="AIza… (from aistudio.google.com/apikey)")
            go = gr.Button("Annotate", variant="primary")
        with gr.Column(scale=2):
            summary = gr.Markdown("")
            table = gr.Dataframe(
                headers=["start", "end", "label", "result", "attributes", "confidence", "flags", "description"],
                datatype=["str"] * 8, label="Subtasks", wrap=True, interactive=False,
            )
            warnings = gr.Textbox(label="Warnings (what the validator changed)", lines=3, interactive=False)
            raw = gr.JSON(label="Full response")
    go.click(run, [video, prompt, subtasks, attributes, quality, key],
             [table, summary, warnings, raw], api_name="annotate")
    ex = _examples()
    if ex:
        gr.Examples(ex, inputs=[video, prompt, subtasks, attributes, quality],
                    label="Examples (DROID, CC-BY-4.0) — add your key, then click Annotate")
    gr.Markdown((HERE / "ABOUT.md").read_text() if (HERE / "ABOUT.md").exists() else "")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")),
        max_file_size=MAX_BYTES,   # refused at upload, before anything touches disk
    )
