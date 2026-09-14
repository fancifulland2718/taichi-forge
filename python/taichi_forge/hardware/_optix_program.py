"""Managed user OptiX programs on the existing provider/context/storage owners.

Shader code is trusted application code. Layouts describe C bytes and resource
uses, not a proof of shader bounds or semantic equivalence. All device pointers,
including indirect SBT references, must use owner-bearing resource fields.
"""

import ctypes as c
from dataclasses import dataclass
import json
import struct
from types import MappingProxyType
import weakref

from taichi_forge.hardware import _optix_program_abi as a
from taichi_forge.hardware._memory import HardwareMemoryComponent, make_memory_report
from taichi_forge.hardware._native_adapter import runtime_generation_matches
from taichi_forge.hardware._optix import (
    OptixInstanceScene,
    OptixProvider,
    OptixTriangleScene,
    _invoke_checked,
    _prepare_storage,
)
from taichi_forge.hardware._optix_parameters import OptixParameterLayout, _integer
from taichi_forge.hardware._shader_artifact import PtxModule, _digest
from taichi_forge.lang._storage_view import describe_storage
from taichi_forge.lang._texture import Texture
from taichi_forge.lang.exception import TaichiRuntimeError


def _name(value, label):
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError(f"{label} must be a nonempty name without NUL")
    return value


def _closed():
    raise TaichiRuntimeError("OptiX prepared launch has been closed or its runtime invalidated")


def _uninitialized():
    raise TaichiRuntimeError("call initialize() once before running an OptiX prepared launch")


@dataclass(frozen=True)
class OptixShaderEntry:
    """Entry point in a zero-based program module slot."""

    module: int
    entry: str

    def __post_init__(self):
        _integer(self.module, "module index")
        _name(self.entry, "shader entry")


@dataclass(frozen=True)
class OptixHitGroup:
    """Triangle closest-hit and/or any-hit entry points; no Python callbacks."""

    closest_hit: OptixShaderEntry | None = None
    any_hit: OptixShaderEntry | None = None

    def __post_init__(self):
        if self.closest_hit is None and self.any_hit is None:
            raise ValueError("a hit group requires closest_hit or any_hit")
        if any(
            value is not None and not isinstance(value, OptixShaderEntry) for value in (self.closest_hit, self.any_hit)
        ):
            raise TypeError("hit group entries must be OptixShaderEntry or None")


@dataclass(frozen=True)
class _Data:
    layout: OptixParameterLayout
    scalar_bytes: bytes
    references: tuple

    def __init__(self, layout, values):
        if not isinstance(layout, OptixParameterLayout):
            raise TypeError("data layout must be OptixParameterLayout")
        references = []

        def resource(field, name):
            references.append((field, _name(name, "resource binding")))
            return 0

        packed = layout._pack(values, resource)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "scalar_bytes", packed)
        object.__setattr__(self, "references", tuple(references))

    def pack(self, addresses):
        packed = bytearray(self.scalar_bytes)
        for field, name in self.references:
            struct.pack_into("<Q", packed, field.offset, addresses[name])
        return bytes(packed)

    def to_dict(self):
        return dict(
            layout=self.layout.to_dict(),
            scalar_bytes=self.scalar_bytes.hex(),
            references={field.name: name for field, name in self.references},
        )


@dataclass(frozen=True, init=False)
class OptixSbtRecord:
    """Named group plus a snapshotted C payload, excluding the native SBT header.

    Resource values are binding names, never integer device addresses. A layout's
    alignment applies to its payload; native OptiX SBT alignment is 16 bytes.
    """

    group: str
    _data: _Data

    def __init__(self, group, *, layout=OptixParameterLayout(0), values=None):
        _name(group, "SBT group")
        if not isinstance(layout, OptixParameterLayout):
            raise TypeError("SBT layout must be OptixParameterLayout")
        if layout.alignment > 16:
            raise ValueError("SBT payload alignment cannot exceed native 16-byte alignment")
        object.__setattr__(self, "group", group)
        object.__setattr__(self, "_data", _Data(layout, {} if values is None else values))

    @property
    def layout(self):
        return self._data.layout

    def to_dict(self):
        return dict(group=self.group, **self._data.to_dict())


class OptixProgram:
    """Modules, named groups and one managed pipeline on an OptixProvider.

    Use external PTX compiled against a compatible OptiX SDK. Forge does not compile
    Taichi functions into raygen/hit shaders. Close prepared launches before closing
    their scenes; closing this program retires its own launches first.
    """

    def __init__(
        self,
        provider,
        modules,
        *,
        raygen,
        miss=None,
        hit_groups=None,
        parameters=OptixParameterLayout(0),
        parameter_name="params",
        payload_count=0,
        attribute_count=2,
        max_trace_depth=1,
        allow_opacity_micromaps=False,
    ):
        self._handle = None
        self._launches = weakref.WeakSet()
        if not isinstance(provider, OptixProvider):
            raise TypeError("program requires an OptixProvider")
        provider._validate_lifetime()
        modules = tuple(modules)
        if not modules or any(not isinstance(module, PtxModule) for module in modules):
            raise TypeError("modules must be a nonempty sequence of PtxModule artifacts")
        if not isinstance(parameters, OptixParameterLayout):
            raise TypeError("parameters must be OptixParameterLayout")
        if parameters.size > 0xFFFFFFFF or parameters.alignment > 256:
            raise ValueError("launch parameters must fit uint32 size and CUDA's 256-byte base alignment")
        _name(parameter_name, "parameter variable")
        if _integer(payload_count, "payload count") > 32:
            raise ValueError("payload count must not exceed 32")
        if not 2 <= _integer(attribute_count, "attribute count") <= 8:
            raise ValueError("attribute count must be between 2 and 8")
        _integer(max_trace_depth, "trace depth")
        if max_trace_depth > 0xFFFFFFFF:
            raise ValueError("trace depth must fit uint32")
        if not isinstance(allow_opacity_micromaps, bool):
            raise TypeError("allow_opacity_micromaps must be bool")
        groups, indices, descriptions = [], {}, []

        def entry(value, prefix):
            if value is None:
                return a.Entry()
            if not isinstance(value, OptixShaderEntry):
                raise TypeError("program group entries must be OptixShaderEntry")
            if value.module >= len(modules) or not value.entry.startswith(prefix):
                raise ValueError("shader entry module or OptiX entry prefix is invalid")
            return a.Entry(value.module, value.entry.encode())

        for kind, members in enumerate((raygen, miss or {}, hit_groups or {})):
            for name, value in sorted(dict(members).items()):
                _name(name, "program group")
                if name in indices:
                    raise ValueError("program group names must be unique across stages")
                if kind == 2:
                    if not isinstance(value, OptixHitGroup):
                        raise TypeError("hit_groups values must be OptixHitGroup")
                    first = entry(value.closest_hit, "__closesthit__")
                    second = entry(value.any_hit, "__anyhit__")
                else:
                    first = entry(value, "__raygen__" if kind == 0 else "__miss__")
                    if not first.name:
                        raise ValueError("raygen/miss groups require an entry")
                    second = a.Entry()
                indices[name] = (len(groups), kind)
                groups.append(a.Group(kind, first, second))
                descriptions.append(
                    dict(
                        name=name,
                        kind=kind,
                        entry=None if not first.name else (first.module_index, first.name.decode()),
                        any_hit=None if not second.name else (second.module_index, second.name.decode()),
                    )
                )
        if not any(kind == 0 for _, kind in indices.values()):
            raise ValueError("program requires at least one raygen group")
        self.provider = provider
        self._runtime_prog = provider._runtime_prog
        self._runtime_generation = provider._runtime_generation
        self._api = a.load_program_api(provider)
        self._parameters = parameters
        self._groups = MappingProxyType(indices)
        facts = dict(
            modules=[module.to_dict() for module in modules],
            groups=descriptions,
            parameters=parameters.to_dict(),
            parameter_name=parameter_name,
            payload_count=payload_count,
            attribute_count=attribute_count,
            max_trace_depth=max_trace_depth,
            allow_opacity_micromaps=allow_opacity_micromaps,
            adapter={
                name: provider.identity[name] for name in ("provider_abi", "optix_abi_version", "build_identity")
            },
        )
        self._manifest_json = json.dumps(facts, sort_keys=True, separators=(",", ":"))
        self._program_id = "optix-program:" + _digest(facts)
        native_modules = (a.Module * len(modules))(*(a.Module(module.code, len(module.code)) for module in modules))
        native_groups = (a.Group * len(groups))(*groups)
        desc = a.ProgramDesc(
            c.sizeof(a.ProgramDesc),
            len(modules),
            native_modules,
            len(groups),
            native_groups,
            parameter_name.encode(),
            parameters.size,
            payload_count,
            attribute_count,
            max_trace_depth,
            int(allow_opacity_micromaps),
        )
        handle = c.c_void_p()
        scope = self._runtime_prog._begin_external_cuda_submission()
        try:
            provider._validate_lifetime()
            _invoke_checked(provider._loaded.api, self._api.create, provider._context, c.byref(desc), c.byref(handle))
            self._handle = handle
            provider._programs.add(self)
        finally:
            del scope

    @property
    def program_id(self):
        return self._program_id

    @property
    def parameters(self):
        return self._parameters

    @property
    def closed(self):
        return self._handle is None

    def to_dict(self):
        return dict(program_id=self.program_id, **json.loads(self._manifest_json))

    def _validate_lifetime(self):
        if self.closed:
            raise TaichiRuntimeError("OptixProgram has been closed")
        self.provider._validate_lifetime()

    def record(self, dimensions, *, raygen, miss=(), hit=(), parameters=None, scenes=None):
        self._validate_lifetime()
        return OptixProgramRecording(
            self, dimensions, raygen=raygen, miss=miss, hit=hit, parameters=parameters, scenes=scenes
        )

    def memory_report(self):
        """Pipeline-private driver memory is unknown, not fabricated from stack settings."""
        resident = not self.closed and runtime_generation_matches(self)
        return make_memory_report(
            "optix_program",
            "cuda",
            (
                HardwareMemoryComponent(
                    "pipeline_driver_state", None, False, "provider_generation", "driver", resident=resident
                ),
            ),
            lifecycle_state="closed" if self.closed else "ready" if resident else "runtime_invalid",
            ownership_scope="program_only_excluding_launches_and_shared_context",
        )

    _graph_provider_memory_report = memory_report

    def close(self):
        if self.closed:
            return
        # Graph lifecycle locks precede the native submission guard. A launch
        # may retire referring Graphs, so do that before taking the guard here.
        for launch in tuple(self._launches):
            launch.close()
        scope = self._runtime_prog._begin_external_cuda_submission()
        try:
            _invoke_checked(self.provider._loaded.api, self._api.destroy, self._handle)
            self._handle = None
        finally:
            del scope

    def __enter__(self):
        self._validate_lifetime()
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Explicit close/reset surface errors; Python finalizers cannot.
            pass


@dataclass(frozen=True, init=False, eq=False)
class OptixProgramRecording:
    """Fixed program/SBT/scenes and named buffer/texture inputs and outputs.

    Scalar values are snapshotted here. Prepare again after changing resource
    bindings; record again after changing dimensions, scalars, SBT or scenes.
    In-place device data/refit updates can reuse an initialized preparation.
    """

    program: OptixProgram
    dimensions: tuple
    raygen: OptixSbtRecord
    miss: tuple
    hit: tuple
    scenes: object
    _parameters: _Data
    _resources: object

    def __init__(self, program, dimensions, *, raygen, miss=(), hit=(), parameters=None, scenes=None):
        if not isinstance(program, OptixProgram):
            raise TypeError("recording requires an OptixProgram")
        program._validate_lifetime()
        if isinstance(dimensions, int):
            dimensions = (dimensions,)
        dimensions = tuple(dimensions)
        if not 1 <= len(dimensions) <= 3:
            raise ValueError("OptiX launch dimensions require one to three extents")
        product = 1
        for extent in dimensions:
            product *= _integer(extent, "launch extent", 1)
        if product > 1 << 30:
            raise ValueError("OptiX launch dimensions exceed 2^30 threads")
        dimensions += (1,) * (3 - len(dimensions))
        raygen = OptixSbtRecord(raygen) if isinstance(raygen, str) else raygen
        miss, hit = tuple(miss), tuple(hit)
        for kind, records in enumerate(((raygen,), miss, hit)):
            for record in records:
                if not isinstance(record, OptixSbtRecord):
                    raise TypeError("SBT entries must be OptixSbtRecord")
                group = program._groups.get(record.group)
                if group is None or group[1] != kind:
                    raise ValueError("SBT record has an unknown group or the wrong shader stage")
        data = _Data(program.parameters, {} if parameters is None else parameters)
        resources = {}
        for item in (data, *(record._data for record in (raygen, *miss, *hit))):
            for field, name in item.references:
                old = resources.get(name)
                if old is not None and old[0] != field.kind:
                    raise ValueError("one binding name cannot designate different resource kinds")
                access = field.access if old is None or old[1] == field.access else "read_write"
                resources[name] = (field.kind, access)
        scenes = dict(scenes or {})
        if set(scenes) != {name for name, (kind, _) in resources.items() if kind == "scene"}:
            raise ValueError("scenes must exactly match scene references in parameters and SBT")
        for scene in scenes.values():
            if (
                not isinstance(scene, (OptixTriangleScene, OptixInstanceScene))
                or scene.provider is not program.provider
            ):
                raise ValueError("all program scenes must belong to the same OptixProvider")
            scene._validate_lifetime()
        for key, value in dict(
            program=program,
            dimensions=dimensions,
            raygen=raygen,
            miss=miss,
            hit=hit,
            scenes=MappingProxyType(scenes),
            _parameters=data,
            _resources=MappingProxyType(resources),
        ).items():
            object.__setattr__(self, key, value)

    @property
    def binding_names(self):
        return tuple(name for name, (kind, _) in self._resources.items() if kind != "scene")

    def prepare(self, bindings):
        """Resolve managed storage and allocate a packet, without uploading or launching."""
        return OptixPreparedLaunch(self, bindings)


class OptixPreparedLaunch:
    """One owner-retaining launch packet with explicit, idempotent initialization.

    ``run()`` is asynchronous and uses Forge's ordered CUDA stream. It performs no
    SBT reconstruction, parameter upload or resource resolution. ``close()`` waits
    for this stream before releasing packet storage. Memory reports do not include
    borrowed application resources, shared AS storage or opaque driver allocations.
    """

    def __init__(self, recording, bindings):
        self._handle = None
        self._call = _uninitialized
        if not isinstance(recording, OptixProgramRecording):
            raise TypeError("preparation requires an OptixProgramRecording")
        self.recording = recording
        self.program = recording.program
        self._runtime_prog = self.program._runtime_prog
        self._runtime_generation = self.program._runtime_generation
        self._storage = None
        self._owners = ()
        self._bindings = MappingProxyType({})
        self._initialized = False
        self._graphs = weakref.WeakSet()
        self._closing = False
        self._memory = None
        self.program._validate_lifetime()
        if set(bindings) != set(recording.binding_names):
            raise ValueError("program bindings must exactly match named buffer/texture references")
        values, descriptions, writable, textures, locations = [], [], [], [], {}
        slots = {}
        for name in recording.binding_names:
            value = bindings[name]
            kind, access = recording._resources[name]
            if kind == "buffer":
                description = describe_storage(value, access="readwrite" if access == "read_write" else access)
                if not description.supported:
                    raise ValueError(
                        f"OptiX buffer {name!r} is not supported dense storage: {description.failure_reason}"
                    )
            key = (kind, id(value))
            if key in slots:
                locations[name] = slots[key]
                if kind == "buffer":
                    writable[slots[key][1]] |= access != "read"
                continue
            if kind == "buffer":
                location = (kind, len(values))
                values.append(value)
                descriptions.append(description)
                writable.append(access != "read")
            else:
                if (
                    not isinstance(value, Texture)
                    or value.tex is None
                    or value._runtime_prog is not self._runtime_prog
                ):
                    raise ValueError("OptiX texture must be a managed texture in this runtime")
                location = (kind, len(textures))
                textures.append(value)
            locations[name] = slots[key] = location
        self._storage, self._owners = _prepare_storage(self.program, values, descriptions, tuple(writable), textures)
        addresses = {
            name: int(self._storage.pointers[index] if kind == "buffer" else self._storage.texture_objects[index])
            for name, (kind, index) in locations.items()
        }
        native_scenes, unique_scenes = [], set()
        for name, scene in recording.scenes.items():
            scene._validate_lifetime()
            native = a.Scene(scene._scene, int(isinstance(scene, OptixInstanceScene)))
            info = a.SceneInfo(c.sizeof(a.SceneInfo))
            _invoke_checked(
                self.program.provider._loaded.api,
                self.program._api.scene_info,
                self.program.provider._context,
                c.byref(native),
                c.byref(info),
            )
            addresses[name] = int(info.traversable)
            if id(scene) not in unique_scenes:
                unique_scenes.add(id(scene))
                native_scenes.append(native)
        payloads = []

        def payload(data):
            packed = data.pack(addresses)
            owner = c.create_string_buffer(packed)
            payloads.append(owner)
            return len(packed), c.addressof(owner)

        def record(item):
            size, pointer = payload(item._data)
            return a.Record(self.program._groups[item.group][0], size, pointer)

        size, pointer = payload(recording._parameters)
        raygen = record(recording.raygen)
        miss = (a.Record * len(recording.miss))(*(record(item) for item in recording.miss))
        hit = (a.Record * len(recording.hit))(*(record(item) for item in recording.hit))
        scenes = (a.Scene * len(native_scenes))(*native_scenes)
        desc = a.LaunchDesc(
            c.sizeof(a.LaunchDesc),
            *recording.dimensions,
            pointer,
            size,
            raygen,
            miss,
            len(miss),
            hit,
            len(hit),
            scenes,
            len(scenes),
        )
        handle = c.c_void_p()
        scope = self._runtime_prog._begin_external_cuda_submission()
        try:
            self.program._validate_lifetime()
            _invoke_checked(
                self.program.provider._loaded.api,
                self.program._api.prepare,
                self.program._handle,
                c.byref(desc),
                c.byref(handle),
            )
            self._handle = handle
            memory = a.Memory(c.sizeof(a.Memory))
            _invoke_checked(self.program.provider._loaded.api, self.program._api.memory, handle, c.byref(memory))
            self._memory = memory
            self._bindings = MappingProxyType(dict(bindings))
            self.program._launches.add(self)
        except BaseException:
            self.close()
            raise
        finally:
            del scope

    @property
    def closed(self):
        return self._handle is None

    def initialize(self):
        """Upload the private parameter/SBT snapshot once, ordered before launches."""
        if self.closed:
            _closed()
        self._runtime_prog._invoke_external_cuda_prepared(self._storage, self._initialize)
        self._call = self._run
        self._initialized = True
        return self

    def _require_initialized(self):
        if self.closed:
            _closed()
        if not self._initialized:
            _uninitialized()

    def graph_recording(self):
        """Reference this initialized packet from a root Graph; bindings stay fixed."""
        from taichi_forge.hardware._optix_program_graph import _PreparedProgramRecording

        return _PreparedProgramRecording(self)

    def _register_graph_owner(self, graph):
        self._require_initialized()
        if self._closing:
            raise TaichiRuntimeError("cannot attach a Graph to a retiring OptiX launch")
        self._graphs.add(graph)

    def _as_graph_native_node(self):
        return self.graph_recording()._as_graph_native_node()

    def _initialize(self):
        _invoke_checked(self.program.provider._loaded.api, self.program._api.initialize, self._handle, 0)

    def _launch(self):
        _invoke_checked(self.program.provider._loaded.api, self.program._api.launch, self._handle, 0)

    def _run(self):
        return self._runtime_prog._invoke_external_cuda_prepared(self._storage, self._launch)

    def run(self):
        return self._call()

    def memory_report(self):
        resident = not self.closed and runtime_generation_matches(self)
        size = int(self._memory.device_data_bytes) if self._memory is not None else 0
        return make_memory_report(
            "optix_program_launch",
            "cuda",
            (
                HardwareMemoryComponent(
                    "parameter_and_sbt_storage", size, True, "provider_generation", "provider", resident=resident
                ),
            ),
            lifecycle_state="closed" if self.closed else "ready" if resident else "runtime_invalid",
            ownership_scope="prepared_launch_only_excluding_borrowed_resources_and_driver_state",
        )

    def preparation_info(self):
        """Cold requested-byte/stack facts; stack configuration is not resident VRAM."""
        if self._memory is None:
            return {}
        return {name: int(getattr(self._memory, name)) for name, _ in a.Memory._fields_[1:]}

    _graph_provider_memory_report = memory_report

    def _graph_provider_memory_dependencies(self):
        return (self.program, *dict.fromkeys(self.recording.scenes.values()))

    def close(self):
        if self.closed:
            return
        self._closing = True
        scope = None
        try:
            # No native guard is held while acquiring Graph lifecycle locks.
            # Graph close completes in-flight work and drops prepared callables.
            for graph in tuple(self._graphs):
                graph.close()
            scope = self._runtime_prog._begin_external_cuda_submission()
            _invoke_checked(self.program.provider._loaded.api, self.program._api.destroy_launch, self._handle)
            self._handle = None
            self._call = _closed
            self._initialized = False
            self._storage = None
            self._owners = ()
            self._bindings = MappingProxyType({})
        finally:
            self._closing = False
            del scope

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
