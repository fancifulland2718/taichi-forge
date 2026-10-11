import array
import os
import shutil
from zipfile import ZipFile

import pytest
import taichi_forge as ti
from taichi_forge._lib import utils as runtime_utils
from tests import test_utils


@pytest.mark.parametrize("artifact_format", ["bc", "ptx", "cubin"])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_repeated_export_preserves_source_and_dense_relocations(tmp_path, artifact_format):
    native = runtime_utils._loaded_native_runtime_path
    if native is None or not hasattr(runtime_utils._native_library_handles[-1], "ti_copy_memory_from_host_ext"):
        pytest.skip("Requires the shared runtime C API")
    cuda = ti.lang.impl.current_cfg().arch == ti.cuda
    if not cuda and artifact_format != "bc":
        pytest.skip("Device artifact formats require CUDA")
    if artifact_format == "cubin" and not (os.environ.get("TI_CUDA_PTXAS_PATH") or shutil.which("ptxas")):
        pytest.skip("External ptxas is required for cubin export")

    # Adding the earlier tree after export changes the compacted ID of value.
    earlier = ti.field(ti.f32, shape=4)
    value = ti.field(ti.f32, shape=4)

    @ti.kernel
    def store(x: ti.f32):
        for i in range(4):
            value[i] = x

    @ti.kernel
    def read(out: ti.types.ndarray(ti.f32, ndim=1)):
        for i in range(4):
            out[i] = value[i]

    @ti.kernel
    def extend(x: ti.f32, out: ti.types.ndarray(ti.f32, ndim=1)):
        for i in range(4):
            earlier[i] = x
            out[i] = earlier[i] + value[i]

    caps = [ti.DeviceCapability.cuda_compute_capability(ti.lang.impl.get_cuda_compute_capability())] if cuda else []
    exported = ti.aot.Module(artifact_format=artifact_format, caps=caps)
    exported.add_kernel(store)
    exported.add_kernel(read)
    first = tmp_path / "first"
    exported.save(first)
    baseline = {path.name: path.read_bytes() for path in first.iterdir() if path.is_file()}
    packed = tmp_path / "second.tcm"
    exported.archive(str(packed))
    with ZipFile(packed) as archive:
        assert {name: archive.read(name) for name in archive.namelist()} == baseline
    exported.add_kernel(extend)
    expanded = tmp_path / "expanded"
    exported.save(expanded)

    from taichi_forge_runtime.aot import Runtime, Scalar

    with Runtime(
        "cuda" if cuda else "cpu", library_path=native, runtime_path=runtime_utils._runtime_bitcode_dir()
    ) as runtime:
        modules = [runtime.load_module(path) for path in (first, packed, expanded)]
        out = runtime.ndarray("f32", 4)
        for number, module in enumerate(modules, 1):
            module.kernel("store")(Scalar("f32", number))
        for number, module in enumerate(modules, 1):
            module.kernel("read")(out)
            values = array.array("f")
            values.frombytes(out.read())
            assert values.tolist() == [number] * 4
        modules[-1].kernel("extend")(Scalar("f32", 5), out)
        values = array.array("f")
        values.frombytes(out.read())
        assert values.tolist() == [8] * 4
