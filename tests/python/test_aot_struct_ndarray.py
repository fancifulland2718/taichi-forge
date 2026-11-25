import json
import tempfile
from contextlib import closing
from pathlib import Path

import numpy as np

import taichi_forge as ti
from tests import test_utils
from tests.cpp.aot.python_scripts.struct_ndarray_aot import export


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_struct_member_transfers_round_trip_through_an_exported_kernel(tmp_path):
    arch = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    record, _, _ = export(arch, tmp_path, initialize=False)
    n = 32768
    array = ti.ndarray(record, n)
    expected = np.empty(n, dtype=array.numpy_dtype)
    expected.view(np.uint8).fill(0xA5)
    expected["tag"] = expected["inner"]["code"] = 0
    expected["inner"]["value"] = expected["velocity"] = expected["basis"] = 0
    array.from_numpy(expected)
    values = np.arange(n, dtype=np.float32) % 103
    velocity = np.zeros((n, 3), dtype=np.float32)
    velocity.view(np.uint32)[:, 0] = 0x7F801234
    velocity[:, 1] = 2
    array.from_numpy_fields({"inner.value": values, "velocity": velocity})
    expected["inner"]["value"], expected["velocity"] = values + 1, velocity

    with ti.aot.borrow_runtime() as runtime, runtime.load_module(tmp_path) as module:
        update = module.kernel("update")
        with closing(runtime.ndarray(update.ndarray_layout(0), n)) as records, closing(runtime.ndarray("f32", n)) as output:
            records.write(array.to_numpy().tobytes())
            with update.prepare(records, output) as call:
                call()
            np.testing.assert_array_equal(np.frombuffer(output.read(), np.float32), values + 2)
            result = records.read()
    assert result == expected.tobytes()
    array.from_numpy(np.frombuffer(result, dtype=array.numpy_dtype))
    fields = array.to_numpy_fields("velocity", "inner.value")
    np.testing.assert_array_equal(fields["velocity"].view(np.uint32), velocity.view(np.uint32))
    np.testing.assert_array_equal(fields["inner.value"], values + 1)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_struct_aot_export_preserves_jit_graph_and_layout():
    arch = {ti.cpu: "cpu", ti.cuda: "cuda", ti.vulkan: "vulkan"}[ti.cfg.arch]
    with tempfile.TemporaryDirectory() as tmp:
        record, update, graph = export(arch, tmp, initialize=False)
        metadata = json.loads((Path(tmp) / "graphs.v2.json").read_text())
        assert metadata["version"] == 2
        records = ti.ndarray(record, 4)
        output = ti.ndarray(ti.f32, 4)
        data = np.zeros(4, dtype=records.numpy_dtype)
        data["inner"]["value"] = np.arange(4, dtype=np.float32)
        records.from_numpy(data)
        update(records, output)
        np.testing.assert_array_equal(output.to_numpy(), np.arange(4))
        graph.run({"records": records, "result": output})
        np.testing.assert_array_equal(output.to_numpy(), np.arange(4) + 1)
        fresh = ti.ndarray(record, 4)
        fresh.from_numpy(data)
        graph.run({"records": fresh, "result": output})
        np.testing.assert_array_equal(output.to_numpy(), np.arange(4))


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_struct_ndarray_template_identity_ignores_extent_not_member_types():
    @ti.kernel
    def read(records: ti.types.ndarray(), output: ti.types.ndarray()):
        for i in records:
            output[i] = records[i].value

    first = ti.types.struct(value=ti.f32, tag=ti.i32)
    second = ti.types.struct(value=ti.f32, tag=ti.u32)
    module = ti.aot.Module()
    with module.add_kernel_template(read) as template:
        template.instantiate(records=ti.ndarray(first, 4), output=ti.ndarray(ti.f32, 4))
        template.instantiate(records=ti.ndarray(first, 7), output=ti.ndarray(ti.f32, 7))
        assert len(template._instantiated_keys) == 1
        template.instantiate(records=ti.ndarray(second, 4), output=ti.ndarray(ti.f32, 4))
        assert len(template._instantiated_keys) == 2
    with tempfile.TemporaryDirectory() as tmp:
        module.save(tmp)
        assert not (Path(tmp) / "graphs.v2.json").exists()
