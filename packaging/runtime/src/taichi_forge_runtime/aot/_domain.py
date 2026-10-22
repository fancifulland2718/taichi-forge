"""Runtime-only execution owners and exact completion tickets."""

import ctypes as C
import operator
import weakref

from . import _c_api as abi


def _unsigned(value, name, *, positive=False):
    value = operator.index(value)
    if not int(positive) <= value <= 0xFFFFFFFF:
        raise ValueError(f"{name} must fit {'positive ' if positive else ''}uint32")
    return value


class Completion:
    """An exact accepted device prefix; wait never holds a runtime lock.

    Dropping or closing the ticket releases its host handle and does not cancel
    work. Runtime/domain ownership retains pending resources independently.
    """

    def __init__(self, api, handle):
        if not handle:
            raise RuntimeError("Native completion creation returned no handle")
        self._api, self._handle = api, handle

    def _native_handle(self):
        handle = self._handle
        if handle is None:
            raise RuntimeError("Completion is closed")
        return handle

    @property
    def info(self):
        info = abi.CompletionInfo()
        self._api.call(self._api.ti_get_completion_info_ext, self._native_handle(), C.byref(info))
        return {name: getattr(info, name) for name, _ in info._fields_}

    def done(self):
        return bool(self._api.call(self._api.ti_query_completion_ext, self._native_handle()))

    @property
    def gpu_timing(self):
        """Nonblocking compute-envelope timing; unavailable durations are None."""
        self._api.load_completion_timing()
        value = abi.CompletionGpuTiming()
        self._api.call(self._api.ti_get_completion_gpu_timing_ext, self._native_handle(), C.byref(value))
        statuses = ("not_sampled", "pending", "ready", "unsupported", "failed")
        return dict(status=statuses[value.status], scope="compute_envelope",
                    owner=value.owner, sequence=value.sequence,
                    duration_ns=value.duration_ns if value.status == 2 else None)

    def wait(self):
        # ctypes releases the GIL; native lookup pins the ticket then releases
        # its mutex before waiting. A concurrent close may win that lookup and
        # produce a stale-handle error, but cannot invalidate a copied ticket.
        self._api.call(self._api.ti_wait_completion_ext, self._native_handle())

    def close(self):
        handle, self._handle = self._handle, None
        if handle is not None:
            self._api.call(self._api.ti_release_completion_ext, handle)

    def __enter__(self):
        self._native_handle()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class CudaExecutionDomain:
    """A bounded CUDA queue owned by one runtime generation.

    Create with Runtime.execution_domain(). Explicit close drains accepted
    work and destroys recordings; driver/provider cleanup may synchronize the
    context. For active scheduling, stop issuing work and wait an exact ticket.
    """

    def __init__(self, runtime, handle):
        if not handle:
            raise RuntimeError("Native execution-domain creation returned no handle")
        self._runtime, self._api, self._handle = runtime, runtime._api, handle
        self._clients = weakref.WeakSet()

    def _native_handle(self):
        handle = self._handle
        if handle is None:
            raise RuntimeError("CUDA execution-domain handle is released")
        return handle

    @property
    def info(self):
        info = abi.CudaExecutionDomainInfo()
        self._api.call(self._api.ti_get_cuda_execution_domain_info_ext, self._native_handle(), C.byref(info))
        return {name: getattr(info, name) for name, _ in info._fields_}

    def latest_completion(self):
        handle = self._api.call(self._api.ti_get_cuda_execution_domain_completion_ext, self._native_handle())
        try:
            return Completion(self._api, handle)
        except BaseException:
            if handle:
                self._api.call(self._api.ti_release_completion_ext, handle)
            raise

    def wait(self):
        with self.latest_completion() as completion:
            completion.wait()

    def collect(self, budget=1):
        return self._api.call(
            self._api.ti_collect_cuda_execution_domain_ext, self._native_handle(), _unsigned(budget, "budget")
        )

    def stop(self):
        """Permanently close admission without waiting for accepted work."""
        self._api.call(self._api.ti_stop_cuda_execution_domain_ext, self._native_handle())

    def close(self):
        if self._handle is not None:
            handle = self._native_handle()
            self._api.call(self._api.ti_close_cuda_execution_domain_ext, handle)
            for runtime in tuple(self._clients):
                with runtime._lock:
                    runtime._domain_owners.pop(handle, None)
            self._clients.clear()
            self.release()

    def release(self):
        """Release this host handle; the runtime retains its bounded resources."""
        handle, self._handle = self._handle, None
        if handle is not None:
            self._api.call(self._api.ti_release_cuda_execution_domain_ext, handle)

    def __enter__(self):
        self._native_handle()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


def create_domain(runtime, capacity, instance_capacity, priority, gpu_timing=False):
    if type(gpu_timing) is not bool:
        raise TypeError("gpu_timing must be bool")
    capacity = _unsigned(capacity, "capacity", positive=True)
    instance_capacity = _unsigned(instance_capacity, "instance_capacity", positive=True)
    priority = operator.index(priority)
    if not -(1 << 31) <= priority < 1 << 31:
        raise ValueError("priority must fit int32")
    with runtime._lock:
        runtime._check()
        if runtime.arch != "cuda":
            raise NotImplementedError("Execution domains require a CUDA runtime")
        runtime._api.load_cuda_domains()
        options = abi.CudaExecutionDomainCreateInfo(capacity, instance_capacity, priority)
        function = runtime._api.ti_create_cuda_execution_domain_ext
        if gpu_timing:
            runtime._api.load_completion_timing()
            function = runtime._api.ti_create_timed_cuda_execution_domain_ext
        handle = runtime._api.call(function, runtime._handle, C.byref(options))
        try:
            return CudaExecutionDomain(runtime, handle)
        except BaseException:
            if handle:
                runtime._api.call(runtime._api.ti_close_cuda_execution_domain_ext, handle)
                runtime._api.call(runtime._api.ti_release_cuda_execution_domain_ext, handle)
            raise
