"""Cold, allocation-independent identities for caller-owned graphics programs.

Arbitrary SPIR-V has no Forge-owned semantic equivalence proof. Its exact content
and fixed state therefore bound semantic reuse conservatively. A provider that
offers equivalent shader implementations owns that higher-level equivalence;
neither the Graph executor nor CompileIQ attempts to infer it.
"""

import hashlib
import json
from dataclasses import asdict


def _digest(facts):
    encoded = json.dumps(facts, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def graphics_pipeline_identity(
    *,
    shaders,
    vertex_bindings=(),
    vertex_attributes=(),
    shader_buffers=(),
    shader_images=(),
    shader_acceleration_structures=(),
    **state,
):
    """Hash once at pipeline creation; retain neither shader bytes nor handles."""
    extra = {}
    if shader_acceleration_structures:
        extra["shader_acceleration_structures"] = tuple(
            asdict(item) for item in sorted(shader_acceleration_structures, key=lambda x: (x.set_index, x.binding))
        )
    return _digest(
        {
            "schema": "taichi_forge.graphics_pipeline.v1",
            "shaders": tuple((stage, hashlib.sha256(code).hexdigest()) for stage, code in shaders),
            "vertex_bindings": tuple(asdict(item) for item in sorted(vertex_bindings, key=lambda x: x.binding)),
            "vertex_attributes": tuple(
                (item.location, item.binding, item.format.name, item.offset)
                for item in sorted(vertex_attributes, key=lambda x: x.location)
            ),
            "shader_buffers": tuple(
                (type(item).__name__, asdict(item))
                for item in sorted(shader_buffers, key=lambda x: (x.set_index, x.binding))
            ),
            "shader_images": tuple(
                asdict(item) for item in sorted(shader_images, key=lambda x: (x.set_index, x.binding))
            ),
            "state": state,
            **extra,
        }
    )


def _draw(
    draw,
    pipeline,
    vertices,
    index,
    buffers=(),
    images=(),
    indirect=None,
    count=None,
    acceleration_structures=(),
):
    return {
        "pipeline": pipeline._graphics_pipeline_id,
        "draw_kind": type(draw).__name__,
        "draw": asdict(draw),
        "vertices": vertices,
        "index": index,
        "shader_buffers": buffers,
        "shader_images": images,
        "indirect": indirect,
        "count": count,
        **({"shader_acceleration_structures": acceleration_structures} if acceleration_structures else {}),
    }


def graphics_recording_identity(recording, *, single_draw=False):
    """Called by the existing native adapter only when compiling a Graph node."""
    if single_draw:
        draws = (
            _draw(
                recording.draw,
                recording.pipeline,
                recording._ordered_vertex_buffers,
                recording.index_buffer,
            ),
        )
        colors = ((recording.color, "clear", "store", recording.clear_color),)
        depth_load, depth_store = "clear", "store"
    else:
        draws = tuple(
            _draw(
                item.draw,
                item.pipeline,
                item._ordered_vertex_buffers,
                item.index_buffer,
                tuple(sorted(item.shader_buffers.items())),
                item._ordered_shader_images,
                item.indirect_buffer,
                item.count_buffer,
                item._ordered_shader_acceleration_structures,
            )
            for item in recording.draws
        )
        colors = tuple(
            (item.name, item.load_op, item.store_op, item.clear_value)
            for item in recording.colors
        )
        depth_load, depth_store = recording.depth_load_op, recording.depth_store_op
    facts = {
        "schema": "taichi_forge.graphics_recording.v1",
        "draws": draws,
        "colors": colors,
        "depth": (
            None
            if recording.depth is None
            else (recording.depth, depth_load, depth_store, recording.clear_depth)
        ),
        "viewport": recording.viewport,
    }
    return (
        "graphics-semantics:" + _digest(facts),
        "graphics-plan:"
        + _digest(
            {
                "recording": facts,
                "retained_replay": recording._experimental_retained_replay,
                "command_kind": "draw" if single_draw else "pass",
            }
        ),
    )
