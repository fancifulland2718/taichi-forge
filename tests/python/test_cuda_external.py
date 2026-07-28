from types import SimpleNamespace

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.interop import cuda_external as ce
from taichi_forge.lang.exception import TaichiRuntimeTypeError
from tests import test_utils


_UUID = bytes(range(16))


class _HandleDesc:
    def __init__(self):
        self.type = None
        self.handle = SimpleNamespace(win32=SimpleNamespace(handle=0, name=0))
        self.size = 0
        self.flags = 0


class _BufferDesc:
    def __init__(self):
        self.offset = 0
        self.size = 0
        self.flags = 0


class _FakeCudaDriver:
    CUDA_EXTERNAL_MEMORY_DEDICATED = 1
    CUexternalMemoryHandleType = SimpleNamespace(CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32=2)
    CUexternalSemaphoreHandleType = SimpleNamespace(CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32=2)
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC = _HandleDesc
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC = _BufferDesc
    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC = _HandleDesc
    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS = SimpleNamespace
    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS = SimpleNamespace

    def __init__(self, *, map_error=False):
        self.calls = []
        self.map_error = map_error

    def cuCtxGetCurrent(self):
        return 0, 101

    def cuCtxGetDevice(self):
        return 0, 0

    def cuDeviceGetUuid(self, device):
        return 0, SimpleNamespace(bytes=_UUID)

    def cuImportExternalMemory(self, desc):
        self.calls.append(("import_memory", desc.handle.win32.handle, desc.size, desc.flags))
        return 0, 201

    def cuExternalMemoryGetMappedBuffer(self, external_memory, desc):
        self.calls.append(("map_memory", external_memory, desc.offset, desc.size))
        if self.map_error:
            return (2,)
        return 0, 0x100000

    def cuDestroyExternalMemory(self, external_memory):
        self.calls.append(("destroy_memory", external_memory))
        return (0,)

    def cuMemFree(self, pointer):
        self.calls.append(("free", int(pointer)))
        return (0,)

    def cuImportExternalSemaphore(self, desc):
        self.calls.append(("import_semaphore", desc.handle.win32.handle))
        return 0, 301

    def cuWaitExternalSemaphoresAsync(self, semaphores, params, count, stream):
        self.calls.append(("wait", tuple(semaphores), count, int(stream)))
        return (0,)

    def cuSignalExternalSemaphoresAsync(self, semaphores, params, count, stream):
        self.calls.append(("signal", tuple(semaphores), count, int(stream)))
        return (0,)

    def cuDestroyExternalSemaphore(self, semaphore):
        self.calls.append(("destroy_semaphore", semaphore))
        return (0,)

    def cuStreamSynchronize(self, stream):
        self.calls.append(("synchronize", int(stream)))
        return (0,)

    def cuCtxPushCurrent(self, context):
        self.calls.append(("push", int(context)))
        return (0,)

    def cuCtxPopCurrent(self):
        self.calls.append(("pop",))
        return 0, 101

    def cuGetErrorName(self, error):
        return 0, b"CUDA_ERROR_TEST"

    def cuGetErrorString(self, error):
        return 0, b"injected failure"


@pytest.fixture
def fake_cuda(monkeypatch):
    driver = _FakeCudaDriver()
    closed_handles = []
    monkeypatch.setattr(ce, "_load_cuda_driver", lambda: driver)
    monkeypatch.setattr(
        ce,
        "_require_taichi_cuda_context",
        lambda unused_driver: (101, 0, _UUID),
    )
    monkeypatch.setattr(ce, "_close_win32_handle", closed_handles.append)
    return driver, closed_handles


def test_memory_typed_offset_views_and_raii(fake_cuda):
    driver, closed_handles = fake_cuda
    memory = ce.import_vulkan_memory_win32(41, allocation_size=512, device_uuid=_UUID)

    header_u32 = memory.array(shape=2, dtype=ti.u32, offset=0)
    header_u64 = memory.array(shape=(1,), dtype=ti.u64, offset=8)
    body_xform = memory.array(shape=(4, 12), dtype=ti.f32, offset=64)

    assert closed_handles == [41]
    assert header_u32.shape == (2,)
    assert header_u32.dtype == np.dtype(np.uint32)
    assert header_u64.data_ptr() == 0x100000 + 8
    assert body_xform.nbytes == 4 * 12 * 4
    assert body_xform.data_ptr() == 0x100000 + 64
    assert ce.is_cuda_external_array(body_xform)
    assert not ce.is_cuda_external_array(
        SimpleNamespace(
            __taichi_cuda_external_array__="taichi-forge.cuda-external-array.v1",
            shape=(4, 12),
            dtype=np.dtype(np.float32),
            nbytes=192,
            data_ptr=lambda: 0x100040,
        )
    )

    with pytest.raises(ValueError, match="aligned"):
        memory.array(shape=1, dtype=ti.f32, offset=65)
    with pytest.raises(ValueError, match="exceeds"):
        memory.array(shape=(200,), dtype=ti.f32, offset=64)

    memory.close()
    assert ("synchronize", 0) in driver.calls
    assert ("free", 0x100000) in driver.calls
    assert ("destroy_memory", 201) in driver.calls
    with pytest.raises(ce.CudaExternalInteropError, match="closed"):
        body_xform.data_ptr()

    call_count = len(driver.calls)
    memory.close()
    assert len(driver.calls) == call_count


def test_binary_semaphore_wait_signal_use_legacy_default_stream(fake_cuda):
    driver, closed_handles = fake_cuda
    semaphore = ce.import_vulkan_semaphore_win32(42, device_uuid=_UUID)

    semaphore.wait()
    semaphore.signal()
    semaphore.close()

    assert closed_handles == [42]
    assert ("wait", (301,), 1, 0) in driver.calls
    assert ("signal", (301,), 1, 0) in driver.calls
    assert ("destroy_semaphore", 301) in driver.calls


def test_mapping_failure_closes_handle_and_rolls_back(monkeypatch):
    driver = _FakeCudaDriver(map_error=True)
    closed_handles = []
    monkeypatch.setattr(ce, "_load_cuda_driver", lambda: driver)
    monkeypatch.setattr(
        ce,
        "_require_taichi_cuda_context",
        lambda unused_driver: (101, 0, _UUID),
    )
    monkeypatch.setattr(ce, "_close_win32_handle", closed_handles.append)

    with pytest.raises(ce.CudaExternalInteropError, match="cuExternalMemoryGetMappedBuffer"):
        ce.import_vulkan_memory_win32(43, allocation_size=256, device_uuid=_UUID)

    assert closed_handles == [43]
    assert ("destroy_memory", 201) in driver.calls
    assert not any(call[0] == "free" for call in driver.calls)


@pytest.mark.parametrize("resource", ["memory", "semaphore"])
def test_close_handle_failure_rolls_back_cuda_import(monkeypatch, resource):
    driver = _FakeCudaDriver()
    monkeypatch.setattr(ce, "_load_cuda_driver", lambda: driver)
    monkeypatch.setattr(
        ce,
        "_require_taichi_cuda_context",
        lambda unused_driver: (101, 0, _UUID),
    )

    def fail_close(handle):
        raise ce.CudaExternalInteropError(f"injected CloseHandle failure for {handle}")

    monkeypatch.setattr(ce, "_close_win32_handle", fail_close)
    with pytest.raises(ce.CudaExternalInteropError, match="CloseHandle failure"):
        if resource == "memory":
            ce.import_vulkan_memory_win32(44, allocation_size=256, device_uuid=_UUID)
        else:
            ce.import_vulkan_semaphore_win32(44, device_uuid=_UUID)

    if resource == "memory":
        assert ("destroy_memory", 201) in driver.calls
        assert not any(call[0] == "map_memory" for call in driver.calls)
    else:
        assert ("destroy_semaphore", 301) in driver.calls


def test_uuid_mismatch_closes_handle_before_import(monkeypatch):
    driver = _FakeCudaDriver()
    closed_handles = []
    monkeypatch.setattr(ce, "_load_cuda_driver", lambda: driver)
    monkeypatch.setattr(
        ce,
        "_require_taichi_cuda_context",
        lambda unused_driver: (101, 0, _UUID),
    )
    monkeypatch.setattr(ce, "_close_win32_handle", closed_handles.append)

    with pytest.raises(ce.CudaExternalInteropError, match="UUID mismatch"):
        ce.import_vulkan_memory_win32(45, allocation_size=256, device_uuid=b"\xff" * 16)

    assert closed_handles == [45]
    assert not any(call[0] == "import_memory" for call in driver.calls)


class _BorrowedTestMemory:
    def __init__(self, pointer):
        self._data_ptr = pointer
        self.closed = False
        self.device_uuid = _UUID

    def _require_open(self):
        if self.closed:
            raise ce.CudaExternalInteropError("CUDA external memory is closed")

    def _require_current_context(self):
        return None


@test_utils.test(arch=ti.cpu)
def test_external_cuda_array_rejected_on_cpu_kernel():
    memory = _BorrowedTestMemory(0x100000)
    array = ce.CudaExternalArray(
        memory,
        shape=(4,),
        dtype=np.dtype(np.int32),
        offset=0,
        token=ce._ARRAY_TOKEN,
    )

    @ti.kernel
    def fill(values: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in values:
            values[i] = i

    with pytest.raises(TaichiRuntimeTypeError, match="only supported.*arch=ti.cuda"):
        fill(array)


@test_utils.test(arch=ti.cpu)
def test_external_cuda_array_rejects_needs_grad_annotation():
    memory = _BorrowedTestMemory(0x100000)
    array = ce.CudaExternalArray(
        memory,
        shape=(4,),
        dtype=np.dtype(np.float32),
        offset=0,
        token=ce._ARRAY_TOKEN,
    )

    @ti.kernel
    def fill(values: ti.types.ndarray(dtype=ti.f32, ndim=1, needs_grad=True)):
        for i in values:
            values[i] = i

    with pytest.raises(TaichiRuntimeTypeError, match="do not support gradients"):
        fill(array)


@test_utils.test(arch=ti.cpu)
def test_external_cuda_array_rejects_explicit_grad_access():
    memory = _BorrowedTestMemory(0x100000)
    array = ce.CudaExternalArray(
        memory,
        shape=(1,),
        dtype=np.dtype(np.float32),
        offset=0,
        token=ce._ARRAY_TOKEN,
    )

    @ti.kernel
    def read_grad(values: ti.types.ndarray(dtype=ti.f32, ndim=1)) -> ti.f32:
        return values.grad[0]

    with pytest.raises(TaichiRuntimeTypeError, match="do not support gradients"):
        read_grad(array)


@test_utils.test(arch=ti.cpu)
def test_external_cuda_array_rejects_backward_launch():
    memory = _BorrowedTestMemory(0x100000)
    array = ce.CudaExternalArray(
        memory,
        shape=(1,),
        dtype=np.dtype(np.float32),
        offset=0,
        token=ce._ARRAY_TOKEN,
    )

    @ti.kernel
    def scale(values: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in values:
            values[i] *= 2.0

    scale(np.ones(1, dtype=np.float32))
    with pytest.raises(TaichiRuntimeTypeError, match="do not support gradients"):
        scale.grad(array)


@test_utils.test(arch=ti.cuda)
def test_external_cuda_array_minimal_kernel_write():
    driver = pytest.importorskip("cuda.bindings.driver")
    context = ce._cuda_call(driver, "cuCtxGetCurrent")
    if int(context) == 0:
        pytest.skip("Taichi CUDA context is not current on this thread")

    n = 7
    record_width = 12
    record_offset = 64
    nbytes = record_offset + n * record_width * np.dtype(np.float32).itemsize
    pointer = ce._cuda_call(driver, "cuMemAlloc", nbytes)
    memory = _BorrowedTestMemory(int(pointer))
    header = ce.CudaExternalArray(
        memory,
        shape=(2,),
        dtype=np.dtype(np.uint32),
        offset=0,
        token=ce._ARRAY_TOKEN,
    )
    records = ce.CudaExternalArray(
        memory,
        shape=(n, record_width),
        dtype=np.dtype(np.float32),
        offset=record_offset,
        token=ce._ARRAY_TOKEN,
    )

    @ti.kernel
    def fill(
        output_header: ti.types.ndarray(dtype=ti.u32, ndim=1),
        output_records: ti.types.ndarray(dtype=ti.f32, ndim=2),
    ):
        output_header[0] = n
        output_header[1] = record_width
        for i, j in output_records:
            output_records[i, j] = i * 100.0 + j

    try:
        fill(header, records)
        ti.sync()
        output = np.empty(nbytes, dtype=np.uint8)
        ce._cuda_call(driver, "cuMemcpyDtoH", output.ctypes.data, pointer, nbytes)
        np.testing.assert_array_equal(output[:8].view(np.uint32), [n, record_width])
        record_output = output[record_offset:].view(np.float32).reshape(n, record_width)
        expected = np.arange(n, dtype=np.float32)[:, None] * 100.0
        expected = expected + np.arange(record_width, dtype=np.float32)[None, :]
        np.testing.assert_array_equal(record_output, expected)
    finally:
        ce._cuda_call(driver, "cuMemFree", pointer)
