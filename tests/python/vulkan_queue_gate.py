"""Deterministic queue-prefix gate for serialized-host Vulkan integration tests."""

import ctypes as C
import os
import threading

from tests.python.aot_external_memory_probe import (
    VkHeaderFlags,
    VkRuntimeInfo,
    VkSemaphoreSignalInfo,
    VkSemaphoreTypeCreateInfo,
    VkSubmit,
    VkTimelineSemaphoreSubmitInfo,
    vk_check,
)


class VulkanQueueGate:
    """Block one real queue until release, without waiting its later tail.

    Only use with serial host submission. The native queue index comes from
    the selected domain, or explicitly use zero for the default compute queue.
    Always close gates before a Window/domain/runtime in a finally block.
    """

    def __init__(self, domain, *, queue_index=None):
        self.domain = domain
        runtime = domain._runtime
        self.info = VkRuntimeInfo()
        export = runtime._api.library.ti_export_vulkan_runtime
        export.argtypes = [C.c_void_p, C.POINTER(VkRuntimeInfo)]
        export.restype = None
        runtime._api.call(export, runtime._handle, C.byref(self.info))
        prototype = C.WINFUNCTYPE if os.name == "nt" else C.CFUNCTYPE
        get_instance = prototype(C.c_void_p, C.c_void_p, C.c_char_p)(
            self.info.get_instance_proc_addr
        )
        get_device = prototype(C.c_void_p, C.c_void_p, C.c_char_p)(
            get_instance(self.info.instance, b"vkGetDeviceProcAddr")
        )

        def function(name, result, *args):
            address = get_device(self.info.device, name.encode())
            if not address:
                raise RuntimeError(f"Vulkan queue gate lacks {name}")
            return prototype(result, *args)(address)

        self.signal = function(
            "vkSignalSemaphore", C.c_int32, C.c_void_p, C.POINTER(VkSemaphoreSignalInfo)
        )
        self.destroy_semaphore = function(
            "vkDestroySemaphore", None, C.c_void_p, C.c_uint64, C.c_void_p
        )
        self.destroy_fence = function(
            "vkDestroyFence", None, C.c_void_p, C.c_uint64, C.c_void_p
        )
        self.wait_fence = function(
            "vkWaitForFences", C.c_int32, C.c_void_p, C.c_uint32,
            C.POINTER(C.c_uint64), C.c_uint32, C.c_uint64
        )
        get_queue = function(
            "vkGetDeviceQueue", None, C.c_void_p, C.c_uint32, C.c_uint32,
            C.POINTER(C.c_void_p)
        )
        create_semaphore = function(
            "vkCreateSemaphore", C.c_int32, C.c_void_p,
            C.POINTER(VkHeaderFlags), C.c_void_p, C.POINTER(C.c_uint64)
        )
        create_fence = function(
            "vkCreateFence", C.c_int32, C.c_void_p,
            C.POINTER(VkHeaderFlags), C.c_void_p, C.POINTER(C.c_uint64)
        )
        submit = function(
            "vkQueueSubmit", C.c_int32, C.c_void_p, C.c_uint32,
            C.POINTER(VkSubmit), C.c_uint64
        )
        self.queue = C.c_void_p()
        self.queue_index = domain.info["queue_index"] if queue_index is None else queue_index
        get_queue(self.info.device, self.info.compute_queue_family_index,
                  self.queue_index, C.byref(self.queue))
        if not self.queue.value:
            raise RuntimeError("Vulkan queue gate requires an existing queue")
        self.semaphore, self.fence = C.c_uint64(), C.c_uint64()
        self.lock = threading.Lock()
        self.released = self.timed_out = self.queued = self.closed = False
        self.timer = None
        try:
            kind = VkSemaphoreTypeCreateInfo(1000207002, None, 1, 0)
            semaphore_info = VkHeaderFlags(9, C.addressof(kind), 0)
            vk_check(create_semaphore(self.info.device, C.byref(semaphore_info),
                                      None, C.byref(self.semaphore)))
            vk_check(create_fence(self.info.device, C.byref(VkHeaderFlags(8)),
                                  None, C.byref(self.fence)))
            waits = (C.c_uint64 * 1)(self.semaphore.value)
            values = (C.c_uint64 * 1)(1)
            stages = (C.c_uint32 * 1)(0x10000)  # ALL_COMMANDS
            timeline = VkTimelineSemaphoreSubmitInfo(1000207003, None, 1, values, 0, None)
            submission = VkSubmit(4, C.addressof(timeline), 1, waits, stages,
                                  0, None, 0, None)
            vk_check(submit(self.queue, 1, C.byref(submission), self.fence))
            self.queued = True
            self.timer = threading.Timer(5, self._timeout)
            self.timer.daemon = True
            self.timer.start()
        except BaseException:
            self.close()
            raise

    def _release_locked(self):
        if self.semaphore.value and not self.released:
            signal = VkSemaphoreSignalInfo(1000207005, None, self.semaphore.value, 1)
            vk_check(self.signal(self.info.device, C.byref(signal)))
            self.released = True

    def _timeout(self):
        with self.lock:
            if not self.released:
                self.timed_out = True
                self._release_locked()

    def release(self):
        with self.lock:
            self._release_locked()

    def close(self):
        if self.closed:
            return
        self.release()
        if self.timer is not None:
            self.timer.cancel()
            if self.timer.ident is not None:
                self.timer.join()
        if self.queued:
            vk_check(self.wait_fence(self.info.device, 1, C.byref(self.fence),
                                     1, 0xFFFFFFFFFFFFFFFF))
        if self.fence.value:
            self.destroy_fence(self.info.device, self.fence, None)
            self.fence.value = 0
        if self.semaphore.value:
            self.destroy_semaphore(self.info.device, self.semaphore, None)
            self.semaphore.value = 0
        self.closed = True
