"""Cold sampled-image metadata and explicit consumer seeds for Graph export."""

from taichi_forge._lib import core
from taichi_forge.lang import impl
from taichi_forge.lang._texture import Texture
from taichi_forge.lang.exception import TaichiRuntimeError


class SourceImageBindings:
    def __init__(self, bindings, fixed_bindings):
        self.owners = {
            name: owner
            for name, owner in fixed_bindings.items()
            if type(owner) is Texture
        }
        self.used = set()
        for name, owner in bindings.items():
            if not isinstance(name, str) or not name or "\0" in name:
                raise ValueError(
                    "Source binding names must be nonempty strings without NUL"
                )
            if name in fixed_bindings and fixed_bindings[name] is not owner:
                raise ValueError("Source binding conflicts with private Graph storage")
            self.owners[name] = owner
        for owner in self.owners.values():
            self._validate_owner(owner)

    @staticmethod
    def _validate_owner(owner):
        if owner.tex is None or owner._runtime_prog is not impl.get_runtime().prog:
            raise TaichiRuntimeError(
                "Source texture belongs to a retired or different runtime"
            )

    def require(self, name):
        owner = self.owners.get(name)
        if owner is None:
            raise TaichiRuntimeError(
                "AOT sampled image needs its Texture in source_bindings"
            )
        self._validate_owner(owner)
        return owner

    def resource(self, context, name):
        from taichi_forge.hardware._sampling import SamplerConfig

        owner = self.require(name)
        fmt, shape, mip_levels, sampler = owner._creation_metadata
        if context.arch not in (core.Arch.cuda, core.Arch.vulkan) or mip_levels != 1:
            raise TaichiRuntimeError(
                "AOT sampled-image reconstruction requires single-level CUDA/Vulkan textures"
            )
        basic = SamplerConfig(
            min_filter=sampler.min_filter,
            mag_filter=sampler.min_filter,
            address_mode_u=sampler.address_mode_u,
            address_mode_v=sampler.address_mode_v,
            address_mode_w=sampler.address_mode_w,
        )
        if sampler != basic:
            raise TaichiRuntimeError(
                "AOT sampled-image reconstruction requires a basic sampler with matching min/mag filters"
            )

        public = context.public_resource(owner)
        if public is not None:
            self.used.add(name)
            return public

        def build():
            inputs = context.resource_inputs.require(owner, ("pixels",))
            resource = context.resource(
                owner,
                "image",
                "sampled",
                parameters=dict(
                    format=fmt.name,
                    shape=list(shape),
                    filter=sampler.min_filter,
                    address_modes=[
                        sampler.address_mode_u,
                        sampler.address_mode_v,
                        sampler.address_mode_w,
                    ][: len(shape)],
                ),
            )
            context.initialize_native_once(
                resource,
                "image",
                "upload",
                bindings=dict(image=resource, pixels=inputs["pixels"]),
            )
            return resource

        resource = context.resource_once(
            owner, build, policy="sampled_image_consumer_seed"
        )
        self.used.add(name)
        return resource

    def kernel_resource(self, context, arg):
        owner = self.require(arg.name)
        if len(owner._creation_metadata[1]) != len(arg.texture_shape):
            raise TaichiRuntimeError(
                "AOT sampled Texture dimension differs from the Graph kernel argument"
            )
        return self.resource(context, arg.name)

    def validate_used(self):
        if self.owners.keys() - self.used:
            raise ValueError(
                "source_bindings contains textures unused by the exported Graph"
            )
