"""Fixed SNode states owned independently of a shared loaded code module."""
import ctypes as C
import json

from ._runtime import NativeResource


class SNodeInstance(NativeResource):
    """Empty fixed-tree state; numerical initialization uses existing entries.

    CPU prepared calls and CUDA independent domains bind private state. Allocate
    CUDA instances before preparing sparse views. Shape/tree schema
    stay fixed; active state and values belong to the instance.
    """

    @classmethod
    def create(cls, module, *, plan=None):
        from ._plan import ExecutionPlan

        if plan is not None and not isinstance(plan, ExecutionPlan):
            raise TypeError("SNode instance selection requires an ExecutionPlan")
        runtime = module._runtime
        with runtime._lock:
            module._check()
            if runtime.arch not in ("cpu", "x64", "arm64", "cuda"):
                raise NotImplementedError("SNode instances require CPU/CUDA")
            runtime._api.load_snode_instances(runtime.arch)
            if plan is None:
                handle = runtime._api.call(
                    runtime._api.ti_create_aot_snode_instance_ext, runtime._handle, module._handle)
            else:
                runtime._api.load_snode_instance_selection()
                handle = runtime._api.call(
                    runtime._api.ti_create_aot_snode_instance_for_plan_ext,
                    runtime._handle, module._handle, len(plan._json), plan._json)
            try:
                result = cls(runtime, handle, (module,))
                result._module = module
                return result
            except BaseException:
                runtime._api.call(runtime._api.ti_destroy_native_resource_ext, runtime._handle, handle)
                raise

    def _for_module(self, module):
        self._check()
        if module is not self._module:
            raise ValueError("SNode instance belongs to another loaded module")
        return self._handle

    @property
    def info(self):
        """Read actual code/state tree mappings and generations, never mutate them."""
        runtime = self._runtime
        with runtime._lock:
            self._check()
            query = runtime._api.ti_get_aot_snode_instance_info_ext
            size = C.c_uint64()
            runtime._api.call(query, runtime._handle, self._handle, C.byref(size), None)
            data = C.create_string_buffer(size.value)
            runtime._api.call(query, runtime._handle, self._handle, C.byref(size), data)
            return json.loads(data.value)
