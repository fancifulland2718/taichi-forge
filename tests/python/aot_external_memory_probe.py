"""Source-free CUDA AOT consumer of a minimal real Vulkan binary protocol."""

import array
import ctypes as C
import importlib.abc
import os
import sys
import time
import threading


class NoFrontend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("taichi", "taichi_forge", "numpy"):
            raise AssertionError(f"External AOT consumer imported {fullname}")


class VkRuntimeInfo(C.Structure):
    _fields_ = [
        ("get_instance_proc_addr", C.c_void_p),
        ("api_version", C.c_uint32),
        ("instance", C.c_void_p),
        ("physical_device", C.c_void_p),
        ("device", C.c_void_p),
        ("compute_queue", C.c_void_p),
        ("compute_queue_family_index", C.c_uint32),
        ("graphics_queue", C.c_void_p),
        ("graphics_queue_family_index", C.c_uint32),
    ]


class VkMemoryInfo(C.Structure):
    _fields_ = [
        ("buffer", C.c_uint64),
        ("size", C.c_uint64),
        ("usage", C.c_uint32),
        ("memory", C.c_uint64),
        ("offset", C.c_uint64),
    ]


class VkHeaderFlags(C.Structure):
    _fields_ = [("sType", C.c_uint32), ("pNext", C.c_void_p), ("flags", C.c_uint32)]


class VkHandleInfo(C.Structure):
    _fields_ = [
        ("sType", C.c_uint32),
        ("pNext", C.c_void_p),
        ("object", C.c_uint64),
        ("handleType", C.c_uint32),
    ]


class VkDeviceId(C.Structure):
    _fields_ = [
        ("sType", C.c_uint32),
        ("pNext", C.c_void_p),
        ("deviceUUID", C.c_uint8 * 16),
        ("driverUUID", C.c_uint8 * 16),
        ("deviceLUID", C.c_uint8 * 8),
        ("deviceNodeMask", C.c_uint32),
        ("deviceLUIDValid", C.c_uint32),
    ]


class VkProperties(C.Structure):
    # The test only reads the pNext UUID block; reserve more than the native
    # VkPhysicalDeviceProperties payload with its required uint64 alignment.
    _fields_ = [
        ("sType", C.c_uint32),
        ("pNext", C.c_void_p),
        ("properties", C.c_uint64 * 256),
    ]


class VkSubmit(C.Structure):
    _fields_ = [
        ("sType", C.c_uint32),
        ("pNext", C.c_void_p),
        ("waitSemaphoreCount", C.c_uint32),
        ("pWaitSemaphores", C.POINTER(C.c_uint64)),
        ("pWaitDstStageMask", C.POINTER(C.c_uint32)),
        ("commandBufferCount", C.c_uint32),
        ("pCommandBuffers", C.c_void_p),
        ("signalSemaphoreCount", C.c_uint32),
        ("pSignalSemaphores", C.POINTER(C.c_uint64)),
    ]


class VkCommandPoolInfo(C.Structure):
    _fields_ = [
        ("sType", C.c_uint32),
        ("pNext", C.c_void_p),
        ("flags", C.c_uint32),
        ("queueFamilyIndex", C.c_uint32),
    ]


class VkCommandAllocate(C.Structure):
    _fields_ = [
        ("sType", C.c_uint32),
        ("pNext", C.c_void_p),
        ("commandPool", C.c_uint64),
        ("level", C.c_uint32),
        ("count", C.c_uint32),
    ]


class VkCommandBegin(C.Structure):
    _fields_ = [
        ("sType", C.c_uint32),
        ("pNext", C.c_void_p),
        ("flags", C.c_uint32),
        ("inheritance", C.c_void_p),
    ]


class ReadyGate:
    """Queue the Vulkan signal behind a host event before CUDA issues its wait."""

    def __init__(self, producer):
        self.producer = producer
        self.event, self.pool, self.command = C.c_uint64(), C.c_uint64(), C.c_void_p()
        self.lock = threading.Lock()
        self.released = self.timed_out = False
        self.timer = None
        function = producer.function
        self.set_event = function("vkSetEvent", C.c_int32, C.c_void_p, C.c_uint64)
        self.destroy_event = function(
            "vkDestroyEvent", None, C.c_void_p, C.c_uint64, C.c_void_p
        )
        self.destroy_pool = function(
            "vkDestroyCommandPool", None, C.c_void_p, C.c_uint64, C.c_void_p
        )
        create_event = function(
            "vkCreateEvent",
            C.c_int32,
            C.c_void_p,
            C.POINTER(VkHeaderFlags),
            C.c_void_p,
            C.POINTER(C.c_uint64),
        )
        create_pool = function(
            "vkCreateCommandPool",
            C.c_int32,
            C.c_void_p,
            C.POINTER(VkCommandPoolInfo),
            C.c_void_p,
            C.POINTER(C.c_uint64),
        )
        allocate = function(
            "vkAllocateCommandBuffers",
            C.c_int32,
            C.c_void_p,
            C.POINTER(VkCommandAllocate),
            C.POINTER(C.c_void_p),
        )
        begin = function(
            "vkBeginCommandBuffer", C.c_int32, C.c_void_p, C.POINTER(VkCommandBegin)
        )
        end = function("vkEndCommandBuffer", C.c_int32, C.c_void_p)
        wait = function(
            "vkCmdWaitEvents",
            None,
            C.c_void_p,
            C.c_uint32,
            C.POINTER(C.c_uint64),
            C.c_uint32,
            C.c_uint32,
            C.c_uint32,
            C.c_void_p,
            C.c_uint32,
            C.c_void_p,
            C.c_uint32,
            C.c_void_p,
        )
        device = producer.info.device
        try:
            vk_check(
                create_event(
                    device, C.byref(VkHeaderFlags(10)), None, C.byref(self.event)
                )
            )
            vk_check(
                create_pool(
                    device,
                    C.byref(
                        VkCommandPoolInfo(
                            39, None, 0, producer.info.compute_queue_family_index
                        )
                    ),
                    None,
                    C.byref(self.pool),
                )
            )
            vk_check(
                allocate(
                    device,
                    C.byref(VkCommandAllocate(40, None, self.pool.value, 0, 1)),
                    C.byref(self.command),
                )
            )
            vk_check(begin(self.command, C.byref(VkCommandBegin(42))))
            wait(
                self.command,
                1,
                C.byref(self.event),
                0x4000,
                0x10000,
                0,
                None,
                0,
                None,
                0,
                None,
            )
            vk_check(end(self.command))
            producer.submit(signal=True, command=self.command.value)
            self.timer = threading.Timer(5, self._timeout)
            self.timer.daemon = True
            self.timer.start()
        except BaseException:
            self.close()
            raise

    def _timeout(self):
        self.timed_out = True
        self.release()

    def release(self):
        with self.lock:
            if not self.released and self.event.value:
                vk_check(self.set_event(self.producer.info.device, self.event))
                self.released = True

    def close(self):
        self.release()
        if self.timer is not None:
            self.timer.cancel()
            self.timer.join()
        self.producer.runtime.wait()
        if self.pool.value:
            self.destroy_pool(self.producer.info.device, self.pool, None)
            self.pool.value = 0
        if self.event.value:
            self.destroy_event(self.producer.info.device, self.event, None)
            self.event.value = 0


def vk_check(result):
    if result != 0:
        raise RuntimeError(f"Vulkan producer failed: {result}")


class Producer:
    def __init__(self, runtime):
        from taichi_forge_runtime.aot import _c_api as abi
        from taichi_forge_runtime.aot._runtime import _Memory, NdArray

        self.runtime = runtime
        self.handles, self.semaphores = [], []
        self.memory = None
        api = runtime._api
        self.info = VkRuntimeInfo()
        export = api.library.ti_export_vulkan_runtime
        export.argtypes, export.restype = [C.c_void_p, C.POINTER(VkRuntimeInfo)], None
        api.call(export, runtime._handle, C.byref(self.info))
        prototype = C.WINFUNCTYPE if os.name == "nt" else C.CFUNCTYPE
        get_instance = prototype(C.c_void_p, C.c_void_p, C.c_char_p)(
            self.info.get_instance_proc_addr
        )
        get_device = prototype(C.c_void_p, C.c_void_p, C.c_char_p)(
            get_instance(self.info.instance, b"vkGetDeviceProcAddr")
        )

        def function(name, result, *args, instance=False):
            address = (
                get_instance(self.info.instance, name.encode())
                if instance
                else get_device(self.info.device, name.encode())
            )
            if not address:
                raise RuntimeError(f"Vulkan producer lacks {name}")
            return prototype(result, *args)(address)

        self.function = function
        self.submit_native = function(
            "vkQueueSubmit",
            C.c_int32,
            C.c_void_p,
            C.c_uint32,
            C.POINTER(VkSubmit),
            C.c_uint64,
        )
        self.destroy_semaphore = function(
            "vkDestroySemaphore", None, C.c_void_p, C.c_uint64, C.c_void_p
        )
        identifier = VkDeviceId(1000071004)
        properties = VkProperties(1000059001, C.addressof(identifier))
        query = function(
            "vkGetPhysicalDeviceProperties2",
            None,
            C.c_void_p,
            C.POINTER(VkProperties),
            instance=True,
        )
        query(self.info.physical_device, C.byref(properties))
        self.uuid = bytes(identifier.deviceUUID)
        self.create_semaphore = function(
            "vkCreateSemaphore",
            C.c_int32,
            C.c_void_p,
            C.POINTER(VkHeaderFlags),
            C.c_void_p,
            C.POINTER(C.c_uint64),
        )
        win = os.name == "nt"
        handle = C.c_void_p if win else C.c_int32
        self.handle_bit = 2 if win else 1
        self.export_memory = function(
            "vkGetMemoryWin32HandleKHR" if win else "vkGetMemoryFdKHR",
            C.c_int32,
            C.c_void_p,
            C.POINTER(VkHandleInfo),
            C.POINTER(handle),
        )
        self.export_semaphore = function(
            "vkGetSemaphoreWin32HandleKHR" if win else "vkGetSemaphoreFdKHR",
            C.c_int32,
            C.c_void_p,
            C.POINTER(VkHandleInfo),
            C.POINTER(handle),
        )
        self.handle_class = handle
        try:
            allocation = abi.MemoryAllocateInfo(66 * 4, False, False, True, 1)
            token = api.call(
                api.ti_allocate_memory, runtime._handle, C.byref(allocation)
            )
            self.memory = _Memory._from_import(runtime, token, 66 * 4)
            self.array = NdArray(self.memory, "f32", 1, (66,), ())
            self.array.write(bytes(66 * 4))
            description = VkMemoryInfo()
            export = api.library.ti_export_vulkan_memory
            export.argtypes, export.restype = [
                C.c_void_p,
                C.c_void_p,
                C.POINTER(VkMemoryInfo),
            ], None
            api.call(export, runtime._handle, token, C.byref(description))
            assert description.offset == 0
            self.allocation_size = description.size
            self.memory_handle = self.export_handle(description.memory, memory=True)
            export_info = VkHeaderFlags(1000077000, None, self.handle_bit)
            create_info = VkHeaderFlags(9, C.addressof(export_info), 0)
            for _ in range(2):
                semaphore = C.c_uint64()
                vk_check(
                    self.create_semaphore(
                        self.info.device, C.byref(create_info), None, C.byref(semaphore)
                    )
                )
                self.semaphores.append(semaphore.value)
                self.export_handle(semaphore.value, memory=False)
        except BaseException:
            self.close()
            raise

    def export_handle(self, obj, *, memory):
        win = os.name == "nt"
        kind = (
            (1000073003 if win else 1000074002)
            if memory
            else (1000078003 if win else 1000079001)
        )
        description = VkHandleInfo(kind, None, obj, self.handle_bit)
        handle = self.handle_class()
        vk_check(
            (self.export_memory if memory else self.export_semaphore)(
                self.info.device, C.byref(description), C.byref(handle)
            )
        )
        self.handles.append(handle.value)
        return handle.value

    def submit(self, *, wait=False, signal=False, command=None):
        waits = (C.c_uint64 * 1)(self.semaphores[1])
        signals = (C.c_uint64 * 1)(self.semaphores[0])
        stages = (C.c_uint32 * 1)(0x10000)  # ALL_COMMANDS
        commands = (C.c_void_p * 1)(command)
        info = VkSubmit(
            4,
            None,
            int(wait),
            waits if wait else None,
            stages if wait else None,
            int(command is not None),
            C.cast(commands, C.c_void_p) if command is not None else None,
            int(signal),
            signals if signal else None,
        )
        vk_check(self.submit_native(self.info.compute_queue, 1, C.byref(info), 0))

    def assert_handles_live(self):
        if os.name == "nt":
            query = C.WinDLL("kernel32", use_last_error=True).GetHandleInformation
            query.argtypes, query.restype = [C.c_void_p, C.POINTER(C.c_uint32)], C.c_int
            for handle in self.handles:
                assert query(handle, C.byref(C.c_uint32())), C.get_last_error()
        else:
            for handle in self.handles:
                os.fstat(handle)

    def close(self):
        self.runtime.wait()
        for semaphore in self.semaphores:
            self.destroy_semaphore(self.info.device, semaphore, None)
        self.semaphores.clear()
        if os.name == "nt":
            close = C.WinDLL("kernel32").CloseHandle
            close.argtypes, close.restype = [C.c_void_p], C.c_int
        else:
            close = os.close
        for handle in self.handles:
            close(handle)
        self.handles.clear()
        if self.memory is not None:
            self.memory.close()
            self.memory = None


def check(folder):
    from taichi_forge_runtime.aot import AotError, Runtime, Scalar

    index = int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0"))
    # Independent Vulkan queues too: the held producer signal must not precede
    # the peer's consumer on the same Vulkan queue.
    with (
        Runtime("vulkan", device_index=index) as vk,
        Runtime("vulkan", device_index=index) as peer_vk,
    ):
        producer = Producer(vk)
        peer_producer = None
        try:
            peer_producer = Producer(peer_vk)
            with Runtime("cuda") as runtime:
                imported = dict(
                    memory_handle=producer.handles[0],
                    ready_for_cuda=producer.handles[1],
                    ready_for_external=producer.handles[2],
                    allocation_size=producer.allocation_size,
                    device_uuid=producer.uuid,
                    dtype="f32",
                    shape=64,
                    byte_offset=4,
                )
                try:
                    runtime.import_cuda_external_memory(
                        **dict(imported, device_uuid=bytes(16))
                    )
                except AotError as error:
                    assert "UUID" in str(error)
                else:
                    raise AssertionError("Wrong physical device was accepted")
                producer.assert_handles_live()
                image = runtime.import_cuda_external_memory(**imported)
                producer.assert_handles_live()
                assert image.nbytes == 64 * 4
                module = runtime.load_module(folder)
                domain = runtime.execution_domain(capacity=1, instance_capacity=4)
                call = module.prepare_execution(
                    "public_stage",
                    dict(out=image, seed=Scalar("f32", 7)),
                    execution_domain=domain,
                )
                call.initialize()
                peer_image = runtime.import_cuda_external_memory(
                    **dict(
                        imported,
                        memory_handle=peer_producer.handles[0],
                        ready_for_cuda=peer_producer.handles[1],
                        ready_for_external=peer_producer.handles[2],
                        allocation_size=peer_producer.allocation_size,
                        device_uuid=peer_producer.uuid,
                    )
                )
                peer = runtime.execution_domain(capacity=1, instance_capacity=4)
                peer_call = module.prepare_execution(
                    "public_stage",
                    dict(out=peer_image, seed=Scalar("f32", 2)),
                    execution_domain=peer,
                )
                peer_call.initialize()
                try:
                    image.read()
                except AotError as error:
                    assert "external storage" in str(error)
                else:
                    raise AssertionError("External storage bypassed its epoch")
                # Hold an actual external-ready wait pending on O. Another
                # domain must complete real device writes before O is released.
                blocked = None
                gate = ReadyGate(producer)
                try:
                    blocked = call.submit()
                    assert blocked is not None and not blocked.done()
                    assert call.submit() is None  # The bounded domain is full.
                    peer_producer.submit(signal=True)
                    independent = peer_call.submit()
                    assert independent is not None
                    deadline = time.monotonic() + 2.0
                    while not independent.done() and time.monotonic() < deadline:
                        time.sleep(0.001)
                    assert (
                        independent.done()
                    ), "Pending external O blocked an independent domain"
                    peer_producer.submit(wait=True)
                    peer_values = array.array("f")
                    peer_values.frombytes(peer_producer.array.read())
                    assert peer_values.tolist() == [0] + [
                        11 * 64 * 65 // 2 + i for i in range(64)
                    ] + [0]
                    assert not blocked.done()
                    assert not gate.timed_out
                    independent.close()
                finally:
                    # Release even after a failed assertion, before Runtime
                    # teardown can drain the pending queue. No spin kernel.
                    gate.close()
                if blocked is not None:
                    producer.submit(wait=True)
                    blocked.wait()
                    blocked.close()
                    producer.runtime.wait()
                for _ in range(8):
                    producer.submit(signal=True)
                    ticket = call.submit()
                    assert ticket is not None
                    producer.submit(wait=True)
                    values = array.array("f")
                    values.frombytes(producer.array.read())
                    assert values.tolist() == [0] + [
                        36 * 64 * 65 // 2 + i for i in range(64)
                    ] + [0]
                    ticket.wait()
                    ticket.close()
                image.close()
                assert domain.info["closed"]
                producer.assert_handles_live()
                call.close()
                domain.close()
                peer_image.close()
                peer_call.close()
                peer.close()
        finally:
            if peer_producer is not None:
                peer_producer.close()
            producer.close()
    print("PUBLIC_AOT_EXTERNAL_MEMORY_OK", flush=True)


if __name__ == "__main__":
    sys.meta_path.insert(0, NoFrontend())
    from cuda_cig_test_owner import configure_test_context
    configure_test_context()
    check(sys.argv[1])
