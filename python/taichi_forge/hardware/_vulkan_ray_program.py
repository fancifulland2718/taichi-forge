"""Managed Vulkan RT programs from explicit, externally compiled SPIR-V.

Shader code is trusted application code. Reflection checks descriptor coverage,
not shader bounds, material semantics or data-dependent SBT indices. Descriptor
resources are owner-bearing; SBT and push-constant bytes contain scalar data,
not unowned device pointers. No compiler or vendor runtime is loaded here.
"""

from dataclasses import asdict, dataclass
import json
from functools import partial
import struct
from types import MappingProxyType
import weakref

from taichi_forge._lib import core
from taichi_forge.hardware._memory import HardwareMemoryComponent, make_memory_report
from taichi_forge.hardware._native_adapter import runtime_generation_matches, validate_runtime_generation
from taichi_forge.hardware._ray import InstanceTLAS
from taichi_forge.hardware._runtime import active_backend
from taichi_forge.hardware._shader_artifact import SpirvShader, _digest
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang._storage_view import describe_storage
from taichi_forge.lang._texture import Texture
from taichi_forge.lang.exception import TaichiRuntimeError


def _name(value):
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError("names must be nonempty strings without NUL")
    return value


def _uint(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= 0xFFFFFFFF:
        raise ValueError(f"{name} must be a uint32 >= {minimum}")
    return value


def _closed():
    raise TaichiRuntimeError("Vulkan ray launch is closed or its runtime was invalidated")


def is_program_available():
    """Independent RT-pipeline capability; does not require inline ray query."""
    program = impl.get_runtime().prog
    probe = getattr(program, "_vulkan_ray_program_available", None)
    return bool(program is not None and active_backend() == "vulkan" and probe is not None and probe())


@dataclass(frozen=True)
class VulkanHitGroup:
    closest_hit: SpirvShader | None = None
    any_hit: SpirvShader | None = None

    def __post_init__(self):
        if self.closest_hit is None and self.any_hit is None:
            raise ValueError("a triangle hit group requires closest_hit and/or any_hit")
        for name in ("closest_hit", "any_hit"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, SpirvShader) or value.stage != name):
                raise TypeError(f"{name} must be a matching SpirvShader")


@dataclass(frozen=True)
class VulkanSbtRecord:
    """Named group and immutable scalar payload, excluding the native handle.

    ShaderRecordBufferKHR/std430 payload bytes follow the native group handle.
    Resource access uses declared descriptors, not addresses encoded in data.
    """

    group: str
    data: bytes = b""

    def __post_init__(self):
        _name(self.group)
        if not isinstance(self.data, (bytes, bytearray, memoryview)):
            raise TypeError("SBT data must be bytes-like scalar payload")
        object.__setattr__(self, "data", bytes(self.data))

    def to_dict(self):
        return dict(group=self.group, data=self.data.hex())


@dataclass(frozen=True)
class VulkanRayBinding:
    """One named shader descriptor. Access describes caller-owned effects.

    Sampled images consume the Texture's sampler and mip chain. ``mip_level``
    selects a storage-image view only. AS and sampler descriptors are read-only.
    """

    set: int
    binding: int
    kind: str = "storage_buffer"
    access: str = "read"
    mip_level: int = 0

    def __post_init__(self):
        _uint(self.set, "descriptor set")
        _uint(self.binding, "descriptor binding")
        _uint(self.mip_level, "mip level")
        if self.kind not in ("storage_buffer", "uniform_buffer", "sampled_image", "storage_image", "scene"):
            raise ValueError("unsupported Vulkan ray descriptor kind")
        if self.access not in ("read", "write", "read_write"):
            raise ValueError("access must be read, write or read_write")
        if self.kind in ("uniform_buffer", "sampled_image", "scene") and self.access != "read":
            raise ValueError("uniform, sampled-image and AS descriptors are read-only")
        if self.kind != "storage_image" and self.mip_level:
            raise ValueError("mip_level is only valid for a storage-image view")


class VulkanRayTracingPipeline:
    """Owner of native raygen/miss/triangle-hit stages on the active Vulkan device.

    Each group is named. Shaders are SpirvShader artifacts compiled explicitly
    by the application. The implementation computes the stack; max_trace_depth
    is a recursion bound, not a stack-byte setting or performance claim.
    """

    graph_runtime_lifetime_check_required = False

    def __init__(self, *, raygen, miss=None, hit_groups=None, max_trace_depth=1, allow_opacity_micromaps=False):
        self._handle = None
        self._launches = weakref.WeakSet()
        if not is_program_available():
            raise TaichiRuntimeError(
                "VulkanRayTracingPipeline requires an initialized runtime with RT-pipeline support"
            )
        _uint(max_trace_depth, "max_trace_depth", 1)
        if not isinstance(allow_opacity_micromaps, bool):
            raise TypeError("allow_opacity_micromaps must be bool")
        shaders, groups, indices, facts, shader_indices = [], [], {}, [], {}

        def shader_index(shader, stage):
            if shader is None:
                return 0xFFFFFFFF
            if not isinstance(shader, SpirvShader) or shader.stage != stage:
                raise TypeError(f"{stage} group requires a matching SpirvShader")
            key = shader.artifact_id
            if key not in shader_indices:
                item = core._VulkanRayProgramShader()
                item.stage = getattr(core._VulkanRayShaderStage, stage)
                item.words = struct.unpack(f"<{len(shader.code) // 4}I", shader.code)
                item.entry = shader.entry_point
                shader_indices[key] = len(shaders)
                shaders.append(item)
            return shader_indices[key]

        for kind, members in (("raygen", raygen), ("miss", miss or {}), ("triangles", hit_groups or {})):
            for name, value in sorted(dict(members).items()):
                _name(name)
                if name in indices:
                    raise ValueError("program group names must be unique across stages")
                group = core._VulkanRayProgramGroup()
                group.kind = getattr(core._VulkanRayGroupKind, kind)
                if kind == "triangles":
                    if not isinstance(value, VulkanHitGroup):
                        raise TypeError("hit_groups values must be VulkanHitGroup")
                    group.closest_hit = shader_index(value.closest_hit, "closest_hit")
                    group.any_hit = shader_index(value.any_hit, "any_hit")
                    code = [None if x is None else x.to_dict() for x in (value.closest_hit, value.any_hit)]
                else:
                    if value is None:
                        raise TypeError("raygen/miss groups require a SpirvShader")
                    group.general = shader_index(value, kind)
                    code = value.to_dict()
                indices[name] = (len(groups), kind)
                groups.append(group)
                facts.append(dict(name=name, kind=kind, shaders=code))
        if not any(kind == "raygen" for _, kind in indices.values()):
            raise ValueError("program requires at least one raygen group")
        facts = dict(groups=facts, max_trace_depth=max_trace_depth, opacity_micromaps=allow_opacity_micromaps)
        self._manifest = json.dumps(facts, sort_keys=True)
        self._program_id = "vulkan-ray-program:" + _digest(facts)
        self._groups = MappingProxyType(indices)
        self._runtime_prog = impl.get_runtime().prog
        self._runtime_generation = int(impl.runtime_generation())
        self._handle = self._runtime_prog._create_vulkan_ray_program(
            shaders, groups, max_trace_depth, allow_opacity_micromaps
        )

    @property
    def program_id(self):
        return self._program_id

    @property
    def closed(self):
        return self._handle is None

    def to_dict(self):
        return dict(program_id=self.program_id, **json.loads(self._manifest))

    def _validate_lifetime(self):
        if self.closed:
            _closed()
        validate_runtime_generation(self, "Vulkan ray program runtime was invalidated")

    def record(self, dimensions, *, raygen, bindings, miss=(), hit=(), push_constants=b""):
        self._validate_lifetime()
        return VulkanProgramRecording(
            self, dimensions, raygen=raygen, bindings=bindings, miss=miss, hit=hit, push_constants=push_constants
        )

    def memory_report(self):
        resident = not self.closed and runtime_generation_matches(self)
        return make_memory_report(
            "vulkan_ray_program",
            "vulkan",
            (
                HardwareMemoryComponent(
                    "driver_pipeline_state", None, False, "provider_generation", "driver", resident=resident
                ),
            ),
            ownership_scope="program_only_excluding_launches",
            lifecycle_state="ready" if resident else "closed",
        )

    _graph_provider_memory_report = memory_report

    def close(self):
        if self.closed:
            return
        for launch in tuple(self._launches):
            launch.close()
        handle, self._handle = self._handle, None
        if runtime_generation_matches(self):
            self._runtime_prog._destroy_vulkan_ray_program(handle)

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


@dataclass(frozen=True, init=False, eq=False)
class VulkanProgramRecording:
    program: VulkanRayTracingPipeline
    dimensions: tuple
    raygen: VulkanSbtRecord
    miss: tuple
    hit: tuple
    bindings: object
    push_constants: bytes

    def __init__(self, program, dimensions, *, raygen, bindings, miss=(), hit=(), push_constants=b""):
        if isinstance(dimensions, int):
            dimensions = (dimensions,)
        dimensions = tuple(dimensions)
        if not 1 <= len(dimensions) <= 3:
            raise ValueError("launch dimensions require one to three extents")
        dimensions = tuple(_uint(x, "launch extent", 1) for x in dimensions)
        raygen = VulkanSbtRecord(raygen) if isinstance(raygen, str) else raygen
        miss, hit = tuple(miss), tuple(hit)
        for kind, records in (("raygen", (raygen,)), ("miss", miss), ("triangles", hit)):
            for record in records:
                if (
                    not isinstance(record, VulkanSbtRecord)
                    or program._groups.get(record.group, (None, None))[1] != kind
                ):
                    raise ValueError("SBT records must name program groups of the corresponding kind")
        bindings = dict(bindings)
        locations = set()
        for name, binding in bindings.items():
            _name(name)
            if not isinstance(binding, VulkanRayBinding):
                raise TypeError("bindings map names to VulkanRayBinding values")
            key = (binding.set, binding.binding)
            if key in locations:
                raise ValueError("descriptor locations must be unique")
            locations.add(key)
        if not isinstance(push_constants, (bytes, bytearray, memoryview)):
            raise TypeError("push_constants must be bytes-like scalar data")
        push_constants = bytes(push_constants)
        if len(push_constants) > 128 or len(push_constants) % 4:
            raise ValueError("push constants must fit 128 bytes and be four-byte aligned")
        for key, value in dict(
            program=program,
            dimensions=dimensions + (1,) * (3 - len(dimensions)),
            raygen=raygen,
            miss=miss,
            hit=hit,
            bindings=MappingProxyType(bindings),
            push_constants=push_constants,
        ).items():
            object.__setattr__(self, key, value)

    @property
    def binding_names(self):
        return tuple(self.bindings)

    def to_dict(self):
        return dict(
            program_id=self.program.program_id,
            dimensions=self.dimensions,
            raygen=self.raygen.to_dict(),
            miss=[x.to_dict() for x in self.miss],
            hit=[x.to_dict() for x in self.hit],
            push_constants=self.push_constants.hex(),
            bindings={name: asdict(binding) for name, binding in self.bindings.items()},
        )

    def prepare(self, bindings):
        """Freeze resources without submitting GPU work. Call initialize() next."""
        return VulkanPreparedLaunch(self, bindings)


def _uninitialized():
    raise TaichiRuntimeError("call initialize() before running a Vulkan prepared launch")


class VulkanPreparedLaunch:
    """Fixed descriptors, dimensions, SBT and scalar parameters, reusable with
    in-place device data updates. Initialization is explicit and asynchronous on
    the existing runtime queue. Rebinding requires a new preparation.
    """

    graph_runtime_lifetime_check_required = False

    def __init__(self, recording, bindings):
        self._packet = None
        self._call = _closed
        self._graphs = weakref.WeakSet()
        self._closing = False
        self._initialized = False
        self.recording = recording
        self.program = recording.program
        self.program._validate_lifetime()
        self._runtime_prog = self.program._runtime_prog
        self._runtime_generation = self.program._runtime_generation
        bindings = dict(bindings)
        if set(bindings) != set(recording.binding_names):
            raise ValueError("bindings must exactly match the recorded descriptor names")
        raw = core._VulkanRayProgramLaunch()
        raw.dimensions = recording.dimensions
        raw.push_constants = list(recording.push_constants)

        def record(value):
            result = core._VulkanRayProgramRecord()
            result.group = self.program._groups[value.group][0]
            result.data = list(value.data)
            return result

        raw.raygen = record(recording.raygen)
        raw.miss = [record(value) for value in recording.miss]
        raw.hit = [record(value) for value in recording.hit]
        buffers, images, scenes, owners = [], [], [], list(bindings.values())
        for name, schema in recording.bindings.items():
            value = bindings[name]
            if schema.kind in ("storage_buffer", "uniform_buffer"):
                description = describe_storage(value)
                if not description.supported:
                    raise TaichiRuntimeError(f"{name} requires dense storage: {description.failure_reason}")
                item = core._VulkanRayProgramBuffer()
                item.storage = description.descriptor
                item.uniform = schema.kind == "uniform_buffer"
                item.writable = schema.access != "read"
                buffers.append(item)
                owners.append(description)
                if isinstance(value, Ndarray):
                    owners.append(value.arr)
            elif schema.kind in ("sampled_image", "storage_image"):
                if not isinstance(value, Texture):
                    raise TypeError(f"{name} requires a managed Texture")
                item = core._VulkanRayProgramImage()
                item.texture = value.tex
                item.storage = schema.kind == "storage_image"
                item.mip_level = schema.mip_level
                images.append(item)
                owners.append(value.tex)
            else:
                if not isinstance(value, InstanceTLAS):
                    raise TypeError(f"{name} requires an InstanceTLAS")
                value._validate_lifetime()
                if value._runtime_prog is not self._runtime_prog:
                    raise TaichiRuntimeError("ray scene belongs to another runtime")
                item = core._VulkanRayProgramAS()
                item.handle = value._handle
                scenes.append(item)
            item.set, item.binding = schema.set, schema.binding
        raw.buffers, raw.images, raw.scenes = buffers, images, scenes
        self._packet = self._runtime_prog._prepare_vulkan_ray_launch(self.program._handle, raw)
        self._owners = tuple(owners)
        self._bindings = MappingProxyType(bindings)
        self._call = _uninitialized
        self.program._launches.add(self)

    @property
    def closed(self):
        return self._packet is None or self._closing

    def initialize(self):
        """Enqueue SBT initialization once; do not wait for the GPU."""
        if self.closed:
            _closed()
        self.program._validate_lifetime()
        self._runtime_prog._initialize_vulkan_ray_launch(self._packet)
        self._initialized = True
        self._call = partial(self._runtime_prog._execute_vulkan_ray_launch, self._packet)
        return self

    def run(self):
        return self._call()

    __call__ = run

    def _require_initialized(self):
        if self.closed:
            _closed()
        self.program._validate_lifetime()
        if not self._initialized:
            _uninitialized()
        if self._runtime_prog._vulkan_ray_launch_info(self._packet)["closed"]:
            _closed()

    def _vulkan_graph_command(self):
        self._require_initialized()
        return self._runtime_prog._vulkan_ray_program_graph_command(self._packet)

    def _register_graph_owner(self, graph):
        self._require_initialized()
        self._graphs.add(graph)

    def graph_recording(self):
        from taichi_forge.hardware._vulkan_ray_program_graph import _PreparedVulkanProgramRecording

        return _PreparedVulkanProgramRecording(self)

    def _as_graph_native_node(self):
        return self.graph_recording()._as_graph_native_node()

    def preparation_info(self):
        if self.closed or not runtime_generation_matches(self):
            return {"closed": 1, "initialized": 0, "sbt_requested_bytes": 0, "upload_requested_bytes": 0}
        return dict(self._runtime_prog._vulkan_ray_launch_info(self._packet))

    def memory_report(self):
        info = self.preparation_info()
        return make_memory_report(
            "vulkan_ray_program",
            "vulkan",
            (
                HardwareMemoryComponent(
                    "sbt",
                    info["sbt_requested_bytes"],
                    True,
                    "provider_generation",
                    "provider",
                    resident=not bool(info["closed"]),
                ),
            ),
            ownership_scope="launch_sbt_only_excluding_upload_staging_and_borrowed_resources",
            lifecycle_state="closed" if info["closed"] else "ready",
        )

    _graph_provider_memory_report = memory_report

    def _graph_provider_memory_dependencies(self):
        return (
            self.program,
            *(value for name, value in self._bindings.items() if self.recording.bindings[name].kind == "scene"),
        )

    def close(self):
        if self.closed:
            return
        self._closing = True
        try:
            for graph in tuple(self._graphs):
                graph.close()
            if runtime_generation_matches(self):
                self._runtime_prog._close_vulkan_ray_launch(self._packet)
            self._packet = None
            self._call = _closed
            self._owners = ()
            self._bindings = MappingProxyType({})
            self._initialized = False
        finally:
            self._closing = False

    def __enter__(self):
        if self.closed:
            _closed()
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
