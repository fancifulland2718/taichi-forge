"""Serialize one immutable command sequence into the shared native registry."""

from taichi_forge._lib import core
from taichi_forge.lang.exception import TaichiRuntimeError


def export(recording, context):
    if context.arch != core.Arch.vulkan:
        raise TaichiRuntimeError("AOT buffer command recordings require Vulkan")
    indices = {name: index for index, name in enumerate(recording.binding_names)}
    commands = []
    for command in recording.commands:
        item = dict(kind=command.kind)
        if command.destination:
            item["destination"] = indices[command.destination]
        if command.kind in ("fill_u32", "copy"):
            item.update(destination_offset=command.destination_offset, bytes=command.bytes)
        if command.kind == "fill_u32":
            item["value"] = command.value
        elif command.kind == "copy":
            item.update(source=indices[command.source], source_offset=command.source_offset)
        commands.append(item)
    context.native(
        "vulkan_rhi", "buffer_commands",
        parameters=dict(buffer_count=len(indices), commands=commands),
        bindings={f"buffer_{index}": name for name, index in indices.items()},
    )
    # The native factory reconstructs finite storage/alias/usage validation.
    context.reconstructed_binding_validators.add(id(recording))
    return ()
