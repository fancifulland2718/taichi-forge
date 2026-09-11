"""Explicit, fixed-resource single-pass downsampling of managed Vulkan images."""

from functools import partial

from taichi_forge.graph._ir import GraphAccess, ResourceEffect
from taichi_forge.graph._native import BackendCommandRecording
from taichi_forge.graph._recipes.definition import _digest
from taichi_forge.hardware._memory import HardwareMemoryComponent, make_memory_report
from taichi_forge.hardware._native_adapter import (
    native_recording_node,
    runtime_generation_matches,
)
from taichi_forge.hardware._runtime import active_backend
from taichi_forge.hardware._spd_jit import compile_shader
from taichi_forge.lang import impl
from taichi_forge.lang._texture import Texture
from taichi_forge.lang.enums import Format
from taichi_forge.lang.exception import TaichiRuntimeError


class VulkanSpdPlan:
    """Prepare standalone FidelityFX SPD against fixed source/output textures.

    Source level zero is read on device. Output level zero is half its size;
    all allocated output levels are written in one dispatch. The supported
    source formats are R32F, RGBA8 and RGBA32F; output is matching-channel FP32.
    Width/height are 2..4096, with no additional 1D tail after the short axis
    reaches one. Odd trailing texels are cropped at each reduction level.
    No gamma conversion or NaN propagation policy is implied.

    ``source_path`` names the official standalone v2.0 ``ffx-spd`` directory;
    ``compiler_path`` names glslangValidator. Both dependencies are explicit.
    Compilation, descriptors, leases and counter initialization happen here,
    not during replay. ``run()`` enqueues work without a host readback or wait.
    """

    graph_runtime_lifetime_check_required = False

    def __init__(self, source, output, *, source_path, compiler_path, reduction="mean"):
        self._closed = True
        program = impl.get_runtime().prog
        if program is None or active_backend() != "vulkan":
            raise TaichiRuntimeError("SPD requires an initialized Vulkan backend")
        if reduction not in ("mean", "min", "max"):
            raise ValueError("SPD reduction must be mean, min or max")
        for image in (source, output):
            if (
                not isinstance(image, Texture)
                or image.tex is None
                or image._runtime_prog is not program
                or image.num_dims != 2
            ):
                raise ValueError("SPD requires live 2D textures from the active runtime")
        if source is output or any(size < 2 or size > 4096 for size in source.shape):
            raise ValueError("SPD requires distinct source/output and source dimensions in 2..4096")
        if output.shape != tuple(size // 2 for size in source.shape):
            raise ValueError("SPD output level zero must be half the source dimensions")
        if output.mip_levels > min(source.shape).bit_length() - 1:
            raise ValueError("SPD output levels cannot extend past the source's shorter axis")
        formats = {
            Format.r32f: "r32f",
            Format.rgba8: "rgba8",
            Format.rgba32f: "rgba32f",
        }
        expected_output = Format.r32f if source.fmt == Format.r32f else Format.rgba32f
        if source.fmt not in formats or output.fmt != expected_output:
            raise ValueError("SPD supports R32F to R32F or RGBA8/RGBA32F to RGBA32F")
        create = getattr(program, "_create_vulkan_spd_plan", None)
        if create is None:
            raise TaichiRuntimeError("SPD native adapter is unavailable in this runtime build")
        shader, facts = compile_shader(
            compiler_path,
            source_path,
            formats[source.fmt],
            formats[output.fmt],
            output.mip_levels,
            reduction,
        )
        self._runtime_prog = program
        self._runtime_generation = int(impl.runtime_generation())
        self._source, self._output = source, output
        self._handle = create(source.tex, output.tex, shader)
        self._closed = False
        try:
            self._statistics = dict(program._vulkan_spd_plan_statistics(self._handle))
            self._facts = facts
            self._reduction = reduction
            self._semantic_id = _digest(
                (
                    "image-reduction-pyramid-v1",
                    source.shape,
                    formats[source.fmt],
                    formats[output.fmt],
                    output.mip_levels,
                    reduction,
                    "floor-crop",
                )
            )
            # Paths and live allocations are observations, not physical strategy.
            self._physical_id = _digest((self._semantic_id, facts["shader_sha256"], self._statistics))
            self._submit = partial(program._vulkan_spd_execute, self._handle)
        except BaseException:
            self.close()
            raise

    closed = property(lambda self: self._closed)
    output = property(lambda self: self._output)

    def run(self):
        self._submit()

    def validate_graph_lifetime(self):
        if self.closed or not runtime_generation_matches(self):
            raise TaichiRuntimeError("SPD plan is closed or belongs to a previous runtime generation")

    def record(self, *, source="source", output="output"):
        self.validate_graph_lifetime()
        if not all(isinstance(name, str) and name for name in (source, output)) or source == output:
            raise ValueError("SPD binding names must be distinct nonempty strings")
        return _Recording(self, (source, output)).as_node()

    def statistics(self):
        return {
            **self._statistics,
            **self._facts,
            "reduction": self._reduction,
            "physical_plan_id": self._physical_id,
            "graph_integration": "root_ordered",
            "gpu_sequence": "retained_secondary_commands",
            "compileiq_search": "complete_recipe_only_no_raw_provider_axis",
        }

    def memory_report(self):
        valid = runtime_generation_matches(self)
        return make_memory_report(
            "fidelityfx_spd",
            "vulkan",
            (
                HardwareMemoryComponent(
                    "counter",
                    4,
                    True,
                    "provider_generation",
                    "provider",
                    resident=valid and not self.closed,
                ),
                HardwareMemoryComponent(
                    "driver_objects_and_pending_commands",
                    None,
                    False,
                    "provider_generation",
                    "driver",
                    resident=valid,
                ),
            ),
            lifecycle_state=("closed" if self.closed else "ready" if valid else "runtime_invalid"),
            ownership_scope="counter excludes caller textures and opaque driver allocations; close is not a retirement observation",
        )

    _graph_provider_memory_report = memory_report

    def _graph_provider_memory_identity(self):
        return ("fidelityfx_spd", self._runtime_generation, self._handle)

    def close(self):
        if not self.closed:
            self._runtime_prog._destroy_vulkan_spd_plan(self._handle)
            self._closed = True

    def __enter__(self):
        self.validate_graph_lifetime()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def __del__(self):
        if not getattr(self, "_closed", True):
            self.close()


class _Recording(BackendCommandRecording):
    def __init__(self, plan, names):
        super().__init__(
            backend="vulkan",
            binding_names=names,
            command_count=1,
            workspace_ownership="provider_generation",
            replay_mode="native_replay",
        )
        object.__setattr__(self, "plan", plan)
        object.__setattr__(self, "_graph_semantic_fingerprint", plan._semantic_id)
        object.__setattr__(self, "_graph_physical_plan_id", plan._physical_id)

    source = property(lambda self: self.plan)

    def _vulkan_graph_command(self):
        self.plan.validate_graph_lifetime()
        return self.plan._runtime_prog._vulkan_spd_graph_command(self.plan._handle, *self.binding_names)

    @property
    def resource_effects(self):
        return (
            ResourceEffect(self.binding_names[0], GraphAccess.READ, subresource=("image", "whole")),
            ResourceEffect(self.binding_names[1], GraphAccess.WRITE, subresource=("image", "whole")),
        )

    def validate_graph_bindings(self, bindings):
        if (
            bindings[self.binding_names[0]] is not self.plan._source
            or bindings[self.binding_names[1]] is not self.plan._output
        ):
            raise TaichiRuntimeError("SPD Graph requires its original texture bindings")

    def execute(self, bindings):
        self.plan.run()

    def as_node(self):
        return native_recording_node(
            self,
            runtime_bindings=tuple((name, "texture") for name in self.binding_names),
            lifetime_leases=(self.plan,),
            debug_info={
                "kind": "fidelityfx_spd",
                "graph_integration": "root_ordered",
                "gpu_sequence": "retained_secondary_commands",
            },
            publish_time_binding_validation_stable=True,
        )
