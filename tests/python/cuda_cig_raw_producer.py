"""Minimal Vulkan owner-side producer; no Forge Vulkan runtime or allocator."""
import ctypes as C

from aot_external_memory_probe import (
    VkRuntimeInfo, VkHeaderFlags, VkHandleInfo,
    VkCommandPoolInfo, VkCommandAllocate, VkCommandBegin, VkSubmit, vk_check,
)


class BufferInfo(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("flags", C.c_uint32),
                ("size", C.c_uint64), ("usage", C.c_uint32), ("sharing", C.c_uint32),
                ("count", C.c_uint32), ("families", C.c_void_p)]


class MemoryRequirements(C.Structure):
    _fields_ = [("size", C.c_uint64), ("alignment", C.c_uint64), ("types", C.c_uint32)]


class MemoryType(C.Structure):
    _fields_ = [("flags", C.c_uint32), ("heap", C.c_uint32)]


class MemoryHeap(C.Structure):
    _fields_ = [("size", C.c_uint64), ("flags", C.c_uint32)]


class MemoryProperties(C.Structure):
    _fields_ = [("type_count", C.c_uint32), ("types", MemoryType * 32),
                ("heap_count", C.c_uint32), ("heaps", MemoryHeap * 16)]


class AllocationInfo(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p),
                ("size", C.c_uint64), ("memory_type", C.c_uint32)]


class BufferBarrier(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p),
                ("source_access", C.c_uint32), ("destination_access", C.c_uint32),
                ("source_family", C.c_uint32), ("destination_family", C.c_uint32),
                ("buffer", C.c_uint64), ("offset", C.c_uint64), ("size", C.c_uint64)]


class RawProducer:
    def __init__(self, owner):
        from taichi_forge_runtime.aot._cuda_context import _Config
        self.owner = owner
        self.info = C.cast(owner.library.cig_fixture_vulkan(owner.address), C.POINTER(VkRuntimeInfo)).contents
        self.uuid = bytes(C.cast(owner.address, C.POINTER(_Config)).contents.device_uuid)
        self.handles, self.semaphores = [], []
        self.buffer = self.memory = self.pool = self.fence = 0
        self.command = C.c_void_p()
        self.acquire_command = C.c_void_p()
        self.mapped = C.c_void_p()
        prototype = C.WINFUNCTYPE
        get_instance = prototype(C.c_void_p, C.c_void_p, C.c_char_p)(self.info.get_instance_proc_addr)
        get_device = prototype(C.c_void_p, C.c_void_p, C.c_char_p)(get_instance(self.info.instance, b"vkGetDeviceProcAddr"))

        def function(name, result, *args, instance=False):
            address = (get_instance(self.info.instance, name.encode()) if instance
                       else get_device(self.info.device, name.encode()))
            assert address, name
            return prototype(result, *args)(address)

        self.function = function
        device = self.info.device
        self.submit_native = function("vkQueueSubmit", C.c_int32, C.c_void_p, C.c_uint32, C.POINTER(VkSubmit), C.c_uint64)
        self.wait_native = function("vkWaitForFences", C.c_int32, C.c_void_p, C.c_uint32,
                                    C.POINTER(C.c_uint64), C.c_uint32, C.c_uint64)
        self.reset_native = function("vkResetFences", C.c_int32, C.c_void_p, C.c_uint32, C.POINTER(C.c_uint64))
        external = VkHeaderFlags(1000072000, None, 2)  # ExternalMemoryBufferCreateInfo, OPAQUE_WIN32
        buffer = C.c_uint64()
        vk_check(function("vkCreateBuffer", C.c_int32, C.c_void_p, C.POINTER(BufferInfo), C.c_void_p, C.POINTER(C.c_uint64))(
            device, C.byref(BufferInfo(12, C.addressof(external), 0, 66 * 4, 0x23, 0, 0, None)), None, C.byref(buffer)))
        self.buffer = buffer.value
        requirements = MemoryRequirements()
        function("vkGetBufferMemoryRequirements", None, C.c_void_p, C.c_uint64, C.POINTER(MemoryRequirements))(
            device, buffer, C.byref(requirements))
        properties = MemoryProperties()
        function("vkGetPhysicalDeviceMemoryProperties", None, C.c_void_p, C.POINTER(MemoryProperties), instance=True)(
            self.info.physical_device, C.byref(properties))
        memory_type = next(i for i in range(properties.type_count)
                           if requirements.types & (1 << i) and properties.types[i].flags & 6 == 6)
        export = VkHeaderFlags(1000072002, None, 2)  # ExportMemoryAllocateInfo
        memory = C.c_uint64()
        vk_check(function("vkAllocateMemory", C.c_int32, C.c_void_p, C.POINTER(AllocationInfo), C.c_void_p, C.POINTER(C.c_uint64))(
            device, C.byref(AllocationInfo(5, C.addressof(export), requirements.size, memory_type)), None, C.byref(memory)))
        self.memory, self.allocation_size = memory.value, requirements.size
        vk_check(function("vkBindBufferMemory", C.c_int32, C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64)(device, buffer, memory, 0))
        vk_check(function("vkMapMemory", C.c_int32, C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_uint32, C.POINTER(C.c_void_p))(
            device, memory, 0, 66 * 4, 0, C.byref(self.mapped)))
        self.export_handle(memory.value, True)
        external = VkHeaderFlags(1000077000, None, 2)
        for _ in range(2):
            semaphore = C.c_uint64()
            vk_check(function("vkCreateSemaphore", C.c_int32, C.c_void_p, C.POINTER(VkHeaderFlags), C.c_void_p, C.POINTER(C.c_uint64))(
                device, C.byref(VkHeaderFlags(9, C.addressof(external), 0)), None, C.byref(semaphore)))
            self.semaphores.append(semaphore.value)
            self.export_handle(semaphore.value, False)
        pool = C.c_uint64()
        vk_check(function("vkCreateCommandPool", C.c_int32, C.c_void_p, C.POINTER(VkCommandPoolInfo), C.c_void_p, C.POINTER(C.c_uint64))(
            device, C.byref(VkCommandPoolInfo(39, None, 0, self.info.compute_queue_family_index)), None, C.byref(pool)))
        self.pool = pool.value
        vk_check(function("vkAllocateCommandBuffers", C.c_int32, C.c_void_p, C.POINTER(VkCommandAllocate), C.POINTER(C.c_void_p))(
            device, C.byref(VkCommandAllocate(40, None, pool.value, 0, 1)), C.byref(self.command)))
        vk_check(function("vkBeginCommandBuffer", C.c_int32, C.c_void_p, C.POINTER(VkCommandBegin))(
            self.command, C.byref(VkCommandBegin(42, None, 4))))
        function("vkCmdFillBuffer", None, C.c_void_p, C.c_uint64, C.c_uint64, C.c_uint64, C.c_uint32)(
            self.command, self.buffer, 0, 66 * 4, 0)
        barrier = function("vkCmdPipelineBarrier", None, C.c_void_p, C.c_uint32, C.c_uint32,
                           C.c_uint32, C.c_uint32, C.c_void_p, C.c_uint32,
                           C.POINTER(BufferBarrier), C.c_uint32, C.c_void_p)
        # Transfer availability and exclusive ownership to the external API.
        release = BufferBarrier(44, None, 0x1000, 0, self.info.compute_queue_family_index,
                                0xFFFFFFFE, self.buffer, 0, 66 * 4)
        barrier(self.command, 0x1000, 0x2000, 0, 0, None, 1, C.byref(release), 0, None)
        vk_check(function("vkEndCommandBuffer", C.c_int32, C.c_void_p)(self.command))
        vk_check(function("vkAllocateCommandBuffers", C.c_int32, C.c_void_p, C.POINTER(VkCommandAllocate), C.POINTER(C.c_void_p))(
            device, C.byref(VkCommandAllocate(40, None, pool.value, 0, 1)), C.byref(self.acquire_command)))
        vk_check(function("vkBeginCommandBuffer", C.c_int32, C.c_void_p, C.POINTER(VkCommandBegin))(
            self.acquire_command, C.byref(VkCommandBegin(42, None, 4))))
        # The consumed semaphore precedes reacquisition and coherent host read.
        acquire = BufferBarrier(44, None, 0, 0x3000, 0xFFFFFFFE,
                                self.info.compute_queue_family_index, self.buffer, 0, 66 * 4)
        barrier(self.acquire_command, 1, 0x14000, 0, 0, None, 1, C.byref(acquire), 0, None)
        vk_check(function("vkEndCommandBuffer", C.c_int32, C.c_void_p)(self.acquire_command))
        fence = C.c_uint64()
        vk_check(function("vkCreateFence", C.c_int32, C.c_void_p, C.POINTER(VkHeaderFlags), C.c_void_p, C.POINTER(C.c_uint64))(
            device, C.byref(VkHeaderFlags(8, None, 1)), None, C.byref(fence)))
        self.fence = fence.value

    def export_handle(self, value, memory):
        result = C.c_void_p()
        vk_check(self.function("vkGetMemoryWin32HandleKHR" if memory else "vkGetSemaphoreWin32HandleKHR", C.c_int32,
                               C.c_void_p, C.POINTER(VkHandleInfo), C.POINTER(C.c_void_p))(
            self.info.device, C.byref(VkHandleInfo(1000073003 if memory else 1000078003, None, value, 2)), C.byref(result)))
        self.handles.append(result.value)

    def wait(self):
        vk_check(self.wait_native(self.info.device, 1, C.byref(C.c_uint64(self.fence)), 1, 5_000_000_000))

    def submit(self, *, wait=False, signal=False):
        assert wait != signal
        if signal:
            self.wait()
            vk_check(self.reset_native(self.info.device, 1, C.byref(C.c_uint64(self.fence))))
        waits = (C.c_uint64 * 1)(self.semaphores[1])
        signals = (C.c_uint64 * 1)(self.semaphores[0])
        stages = (C.c_uint32 * 1)(0x10000)
        commands = (C.c_void_p * 1)(self.command.value if signal else self.acquire_command.value)
        info = VkSubmit(4, None, int(wait), waits if wait else None, stages if wait else None,
                        1, C.cast(commands, C.c_void_p),
                        int(signal), signals if signal else None)
        vk_check(self.submit_native(self.info.compute_queue, 1, C.byref(info), self.fence if wait else 0))

    def read(self):
        self.wait()
        return C.string_at(self.mapped, 66 * 4)

    def close(self):
        self.wait()
        fn, device = self.function, self.info.device
        fn("vkUnmapMemory", None, C.c_void_p, C.c_uint64)(device, self.memory)
        for name, value in (("vkDestroyCommandPool", self.pool), ("vkDestroyFence", self.fence),
                            *(("vkDestroySemaphore", value) for value in self.semaphores),
                            ("vkDestroyBuffer", self.buffer), ("vkFreeMemory", self.memory)):
            fn(name, None, C.c_void_p, C.c_uint64, C.c_void_p)(device, value, None)
        close = C.WinDLL("kernel32").CloseHandle
        close.argtypes, close.restype = [C.c_void_p], C.c_int
        for handle in self.handles:
            assert close(handle)
