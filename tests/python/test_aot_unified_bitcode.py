import hashlib
import importlib
import json
import sys
import zipfile

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_unified_prepare_forwards_runtime_arrays_through_helpers(tmp_path):
    import numpy as np
    from taichi_forge_runtime.aot import Scalar

    @ti.real_func
    def advance(x: ti.f32, bias: ti.types.ndarray(ti.f32, 1)) -> ti.f32:
        result = x
        for j in range(2):
            result = result * 0.5 + bias[j]
        return result

    @ti.kernel
    def step(count: ti.i32, data: ti.types.ndarray(ti.f32, 1), bias: ti.types.ndarray(ti.f32, 1)):
        for i in range(count):
            data[i] = advance(advance(data[i], bias), bias)

    producer = ti.aot.Module(artifact_format="unified-bc-library")
    producer.add_kernel(step)
    path = tmp_path / "source.tcm"
    producer.archive(str(path))
    with zipfile.ZipFile(path) as archive:
        unit, = json.loads(archive.read("unified_llvm.json"))
        signature, = json.loads(archive.read("unified_abi.json"))
        assert unit["key"] == signature["key"] == "step"

        def read(digest):
            return archive.read(f"code_units/{digest}.bc")

        for count in (12, 5):
            consumer = ti.aot.Module()
            consumer._aot_builder._add_unified_kernel(
                ti.lang.impl.get_runtime().prog, "step", read(unit["value"]),
                archive.read(f'signatures/{signature["value"]}.json').decode(), {(0,): count}, read)
            consumer.archive(str(tmp_path / f"ready-{count}.tcm"))
    with ti.aot.borrow_runtime() as runtime:
        data, bias = runtime.ndarray("f32", (16,)), runtime.ndarray("f32", (2,))
        for count in (12, 5):
            with runtime.load_module(str(tmp_path / f"ready-{count}.tcm")) as module:
                for offset in (0, 7):
                    expected = np.arange(16, dtype=np.float32)
                    values = np.array([0.5, offset + 1], dtype=np.float32)
                    data.write(expected.tobytes())
                    bias.write(values.tobytes())
                    for _ in range(2):
                        for value in values:
                            expected[:count] = expected[:count] * np.float32(0.5) + value
                    with module.kernel("step").bind(Scalar("i32", count), data, bias).prepare() as prepared:
                        prepared.submit().wait()
                    np.testing.assert_array_equal(np.frombuffer(data.read(), np.float32), expected)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_unified_library_retains_independent_dense_bindings(tmp_path):
    left = ti.field(ti.i32)
    right = ti.field(ti.i32)
    for field in (left, right):
        fields = ti.FieldsBuilder()
        fields.dense(ti.i, 8).place(field)
        fields.finalize()

    @ti.kernel
    def update(dst: ti.template()):
        for i in range(8):
            dst[i] = i + 7

    module = ti.aot.Module(ti.lang.impl.current_cfg().arch, artifact_format="unified-bc-library")
    for name, field in (("left", left), ("right", right)):
        module.add_kernel(update, template_args={"dst": field}, name=name)
    path = tmp_path / "fields.tcm"
    module.archive(str(path))
    with zipfile.ZipFile(path) as package:
        entries = {row["key"]: row["value"] for row in json.loads(package.read("unified_llvm.json"))}
        assert entries["left"] == entries["right"]
        code = package.read("code_units/" + entries["left"] + ".bc")
        assert code.startswith(b"BC\xc0\xde")
        assert hashlib.sha256(code).hexdigest() == entries["left"]
        storage = json.loads(package.read("unified_storage.json"))
        left_tree, = storage["entries"]["left"]
        right_tree, = storage["entries"]["right"]
        assert left_tree != right_tree
        layout = storage["trees"][left_tree]
        assert layout == storage["trees"][right_tree]
        assert len(storage["schemas"]) == 1
        assert storage["schemas"][layout]["nodes"][1]["extents"][0]["constant"] == 8
