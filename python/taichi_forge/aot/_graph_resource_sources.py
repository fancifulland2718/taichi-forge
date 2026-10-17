"""Cold typed resource sources shared by kernel and provider Graph export."""

from collections.abc import Mapping

from taichi_forge._lib import core
from taichi_forge.hardware._ray import InstanceTLAS
from taichi_forge.lang._texture import Texture
from taichi_forge.lang.exception import TaichiRuntimeError

from ._graph_image_bindings import SourceImageBindings


class SourceAccelerationStructures:
    def __init__(self, bindings, fixed_bindings):
        self.owners = {name: owner for name, owner in fixed_bindings.items() if type(owner) is InstanceTLAS}
        self.used = set()
        for name, owner in bindings.items():
            if not isinstance(name, str) or not name or "\0" in name:
                raise ValueError("Source binding names must be nonempty strings without NUL")
            if name in fixed_bindings and fixed_bindings[name] is not owner:
                raise ValueError("Source binding conflicts with private Graph storage")
            self.owners[name] = owner
        for owner in self.owners.values():
            owner._validate_lifetime()

    def kernel_resource(self, context, arg):
        from taichi_forge.hardware._aot_vulkan_ray import _scene

        owner = self.owners.get(arg.name)
        if context.arch != core.Arch.vulkan or owner is None:
            raise TaichiRuntimeError("AOT Vulkan AS argument needs its InstanceTLAS in source_bindings")
        resource = _scene(context, owner)
        self.used.add(arg.name)
        return resource

    def validate_used(self):
        if self.owners.keys() - self.used:
            raise ValueError("source_bindings contains acceleration structures unused by the exported Graph")


class SourceResourceBindings:
    """Route qualified source owners without turning them into array bindings."""

    def __init__(self, bindings, fixed_bindings):
        if bindings is None:
            bindings = {}
        if not isinstance(bindings, Mapping):
            raise TypeError("source_bindings must map names to qualified producer arrays or resources")
        handlers = (
            (Texture, core.ArgKind.TEXTURE, SourceImageBindings),
            (InstanceTLAS, core.ArgKind.ACCELERATION_STRUCTURE, SourceAccelerationStructures),
        )
        grouped = {owner_type: {} for owner_type, _, _ in handlers}
        self.arrays = {}
        for name, owner in bindings.items():
            grouped.get(type(owner), self.arrays)[name] = owner
        self._handlers = {tag: factory(grouped[owner_type], fixed_bindings) for owner_type, tag, factory in handlers}
        self.images = self._handlers[core.ArgKind.TEXTURE]
        self.owners = {name: owner for handler in self._handlers.values() for name, owner in handler.owners.items()}

    @property
    def used(self):
        return set().union(*(handler.used for handler in self._handlers.values()))

    def kernel_resource(self, context, arg):
        return self._handlers[arg.tag].kernel_resource(context, arg)

    def validate_used(self):
        for handler in self._handlers.values():
            handler.validate_used()
