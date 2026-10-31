"""Cold fixed-query bindings for the same provider ABI used by native plans.

This is ordered execution, never CUDA stream capture. Immutable parameters
belong to each prepared binding; live rays and per-primitive face values remain
device inputs. Alpha and micromap variants retain their existing adapter path.
"""

import ctypes as c
import weakref

from taichi_forge._lib import core
from taichi_forge._hardware_telemetry import hardware_failure_phase
from taichi_forge.lang.exception import TaichiRuntimeError


class _Description(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("scene_kind", c.c_uint32),
        ("scene", c.c_void_p),
        ("ray_count", c.c_uint32),
        ("occlusion", c.c_uint32),
        ("any_hit", c.c_uint32),
        ("face_count", c.c_uint32),
        ("rays", c.c_uint64),
        ("hits", c.c_uint64),
        ("hit_indices", c.c_uint64),
        ("faces", c.c_uint64),
    ]


class _Memory(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("parameter_bytes", c.c_uint32),
        ("device_bytes", c.c_uint64),
    ]


class _Api(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("revision", c.c_uint32),
        (
            "prepare",
            c.CFUNCTYPE(
                c.c_int, c.c_void_p, c.POINTER(_Description), c.POINTER(c.c_void_p)
            ),
        ),
        ("launch", c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_uint64)),
        ("destroy", c.CFUNCTYPE(c.c_int, c.c_void_p)),
        ("memory", c.CFUNCTYPE(c.c_int, c.c_void_p, c.POINTER(_Memory))),
    ]


def _api(provider):
    from ._optix import _invoke_checked

    if not int(provider._loaded.api.info.features) & (1 << 21):
        return None
    if not hasattr(provider, "_fixed_query_api"):
        api = _Api()
        function = provider._loaded.library.taichi_forge_optix_fixed_query_api
        function.argtypes, function.restype = [
            c.c_uint32,
            c.c_size_t,
            c.POINTER(_Api),
        ], c.c_int
        _invoke_checked(provider._loaded.api, function, 1, c.sizeof(api), c.byref(api))
        if (
            api.struct_size < c.sizeof(api)
            or api.revision != 1
            or not all((api.prepare, api.launch, api.destroy, api.memory))
        ):
            raise TaichiRuntimeError(
                "OptiX fixed-query extension returned an incomplete ABI"
            )
        provider._fixed_query_api = api
    return provider._fixed_query_api


class _PreparedQuery:
    def __init__(self, scene, storage, owners, api, description):
        from ._optix import _invoke_checked

        self.scene, self.storage, self.owners = scene, storage, owners
        self.api, self.handle, self.native = api, c.c_void_p(), None
        self._parameter_bytes = 0
        self._face_bytes = 16 * description.face_count
        self._runtime_prog, self._runtime_generation = (
            scene._runtime_prog,
            scene._runtime_generation,
        )
        loaded = scene.provider._loaded
        try:
            self._runtime_prog._invoke_external_cuda_prepared(
                storage,
                lambda: _invoke_checked(
                    loaded.api,
                    api.prepare,
                    scene.provider._context,
                    c.byref(description),
                    c.byref(self.handle),
                ),
            )
            if not self.handle:
                raise TaichiRuntimeError(
                    "OptiX fixed-query preparation returned a null handle"
                )
            memory = _Memory(c.sizeof(_Memory))
            _invoke_checked(loaded.api, api.memory, self.handle, c.byref(memory))
            self._parameter_bytes = int(memory.device_bytes)
            bind = getattr(core, "_bind_external_cuda_call", None)
            if bind is not None:
                self.native = bind(
                    self._runtime_prog,
                    storage,
                    c.cast(api.launch, c.c_void_p).value,
                    self.handle.value,
                    c.cast(loaded.api.get_last_error, c.c_void_p).value,
                    loaded,
                )
            if not hasattr(scene, "_fixed_queries"):
                scene._fixed_queries = weakref.WeakSet()
            scene._fixed_queries.add(self)
        except BaseException:
            self.close()
            raise

    def __call__(self):
        if not self.handle:
            raise TaichiRuntimeError("OptiX prepared query has been closed")
        self.scene._validate_lifetime()
        with hardware_failure_phase("provider_execution_failure"):
            if self.native is not None:
                return self.native()
            from ._optix import _invoke_checked

            return self._runtime_prog._invoke_external_cuda_prepared(
                self.storage,
                lambda: _invoke_checked(
                    self.scene.provider._loaded.api, self.api.launch, self.handle, 0
                ),
            )

    def close(self):
        from ._optix import _invoke_checked, runtime_generation_matches

        if not self.handle:
            return
        if self.native is not None:
            self.native.invalidate()
            self.native = None
        if runtime_generation_matches(self):
            scope = self._runtime_prog._begin_external_cuda_submission()
            try:
                _invoke_checked(
                    self.scene.provider._loaded.api, self.api.destroy, self.handle
                )
            finally:
                del scope
        self.handle = c.c_void_p()
        self.storage, self.owners = None, ()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _graph_provider_memory_report(self):
        from ._native_adapter import runtime_generation_matches
        from ._memory import HardwareMemoryComponent, make_memory_report

        resident = bool(self.handle) and runtime_generation_matches(self)
        return make_memory_report(
            "optix_fixed_query",
            "cuda",
            (
                HardwareMemoryComponent(
                    "immutable_query_parameters",
                    self._parameter_bytes,
                    True,
                    "provider_generation",
                    "provider",
                    resident=resident,
                ),
                HardwareMemoryComponent(
                    "face_descriptors",
                    self._face_bytes,
                    True,
                    "provider_generation",
                    "runtime",
                    resident=resident,
                ),
            ),
            lifecycle_state="ready" if resident else "closed",
            ownership_scope="prepared_query_excluding_borrowed_arrays_and_scene",
        )


def prepare_fixed_query(recording, bindings, descriptions):
    from ._optix import (
        OptixFaceRuleTable,
        OptixInstanceScene,
        _FaceRule,
        _prepare_storage,
    )
    from taichi_forge.lang._storage_view import describe_storage

    if recording._micromap or any(
        mask is not None for mask in recording.alpha_masks or ()
    ):
        return None
    if recording.hit_indices is None and not recording._occlusion:
        return None
    api = _api(recording.scene.provider)
    if api is None:
        return None
    values = [bindings[recording.rays], bindings[recording.hits]]
    if recording.hit_indices is not None:
        values.append(bindings[recording.hit_indices])
    writable = [False, *([True] * (len(values) - 1))]
    faces, face_descriptions = recording._face_bindings(bindings)
    values.extend(faces)
    descriptions = [*descriptions, *face_descriptions]
    writable.extend([False] * len(faces))
    specs = recording.face_rules or ()
    workspace = None
    if specs:
        import numpy as np
        from taichi_forge.lang._ndarray import ScalarNdarray
        from taichi_forge.types.primitive_types import u64

        workspace = ScalarNdarray(u64, (2 * len(specs),))
        values.append(workspace)
        descriptions.append(describe_storage(workspace))
        writable.append(False)
    storage, owners = _prepare_storage(recording.scene, values, descriptions, writable)
    if specs:
        records = (_FaceRule * len(specs))()
        offset, active = 2 if recording._occlusion else 3, 0
        for i, rule in enumerate(specs):
            if isinstance(rule, OptixFaceRuleTable):
                records[i] = _FaceRule(storage.pointers[offset + active], 2, 0)
                active += 1
            else:
                records[i] = _FaceRule(0, int(rule == "front_only"), 0)
        workspace.from_numpy(np.frombuffer(bytes(records), dtype=np.uint64).copy())
    description = _Description(
        c.sizeof(_Description),
        int(isinstance(recording.scene, OptixInstanceScene)),
        recording.scene._scene,
        recording.ray_count,
        int(recording._occlusion),
        int(recording.any_hit),
        len(specs),
        storage.pointers[0],
        storage.pointers[1],
        0 if recording._occlusion else storage.pointers[2],
        storage.pointers[-1] if specs else 0,
    )
    return _PreparedQuery(recording.scene, storage, owners, api, description)


def close_fixed_queries(scene):
    for query in tuple(getattr(scene, "_fixed_queries", ())):
        query.close()
