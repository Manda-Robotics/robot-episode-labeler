"""Write annotations as an MCAP file.

Robotics datasets keep episode data and annotations in separate MCAP files that
share one absolute time base (this is how XDOF's ABC-130k is laid out), so
`time_origin_ns` is the episode's own start time and segment times are offsets
from it. Written that way, our output drops straight into a team's existing
tooling instead of being another sidecar format they have to adapt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas import AnnotateResponse

TOPIC_SUBTASK = "/subtask_annotation"
TOPIC_INSTRUCTION = "/instruction"

_SUBTASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "start_time": {"type": "number", "description": "Seconds from episode start."},
        "end_time": {"type": "number", "description": "Seconds from episode start."},
        "label": {"type": "string"},
        "result": {"type": "string", "enum": ["pass", "fail", "unknown"]},
        "attributes": {"type": "array", "items": {"type": "string"}},
        "description": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "flags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["start_time", "end_time", "label", "result"],
}

_INSTRUCTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "instruction": {"type": "string"},
        "duration_seconds": {"type": "number"},
        "metadata": {"type": "object"},
    },
    "required": ["instruction"],
}


def write_mcap(
    response: AnnotateResponse,
    path: str | Path,
    time_origin_ns: int = 0,
) -> Path:
    """Write `response` to `path` as MCAP. Returns the path written."""
    from mcap.writer import Writer

    path = Path(path)
    with path.open("wb") as fh:
        writer = Writer(fh)
        writer.start(profile="", library="robot-episode-labeler")

        subtask_schema = writer.register_schema(
            name="rel.SubtaskAnnotation", encoding="jsonschema",
            data=json.dumps(_SUBTASK_SCHEMA).encode(),
        )
        instruction_schema = writer.register_schema(
            name="rel.Instruction", encoding="jsonschema",
            data=json.dumps(_INSTRUCTION_SCHEMA).encode(),
        )
        subtask_channel = writer.register_channel(
            topic=TOPIC_SUBTASK, message_encoding="json", schema_id=subtask_schema,
        )
        instruction_channel = writer.register_channel(
            topic=TOPIC_INSTRUCTION, message_encoding="json", schema_id=instruction_schema,
        )

        writer.add_message(
            instruction_channel, log_time=time_origin_ns, publish_time=time_origin_ns,
            data=json.dumps({
                "instruction": response.task,
                "duration_seconds": response.duration_seconds,
                "metadata": response.metadata,
            }).encode(),
        )

        for segment in response.segments:
            # Logged at the moment the subtask completes, which is where a
            # timeline viewer will show the annotation.
            log_time = time_origin_ns + int(segment.end_seconds * 1e9)
            writer.add_message(
                subtask_channel, log_time=log_time, publish_time=log_time,
                data=json.dumps({
                    "start_time": segment.start_seconds,
                    "end_time": segment.end_seconds,
                    "label": segment.label,
                    "result": segment.result.value,
                    "attributes": segment.attributes,
                    "description": segment.description,
                    "confidence": segment.confidence.value,
                    "flags": segment.flags,
                }).encode(),
            )
        writer.finish()
    return path
