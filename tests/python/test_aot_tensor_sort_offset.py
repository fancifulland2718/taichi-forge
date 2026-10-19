"""Record offsets require scalar alignment, not a whole-record multiple."""

from contextlib import ExitStack, closing
import numpy as np
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_tensor_sort_scalar_aligned_field_offset(tmp_path):
    fields = ti.FieldsBuilder()
    pad, values = ti.field(ti.i32), ti.Vector.field(3, ti.f32)
    fields.dense(ti.i, 1).place(pad)
    fields.dense(ti.i, 17).place(values)
    tree = fields.finalize()
    module = ti.aot.Module()
    module.add_field("values", values, as_array=True)
    module.add_field("pad", pad, as_array=True)
    module.add_execution_plan(
        "sort",
        ti.aot.ExecutionPlan(
            [
                dict(kind="copy", source="input", destination="values"),
                dict(kind="copy", source="guard", destination="pad"),
                dict(kind="sort", keys="keys", values="values"),
                dict(kind="copy", source="values", destination="output"),
                dict(kind="copy", source="pad", destination="guard"),
            ],
            resources=[
                dict(kind="module_array", name=name, view=name)
                for name in ("values", "pad")
            ],
        ),
    )
    artifact = tmp_path / "offset.tcm"
    module.archive(str(artifact))
    with (
        ti.aot.borrow_runtime() as runtime,
        runtime.load_module(artifact) as loaded,
        ExitStack() as stack,
    ):
        arrays = {
            name: stack.enter_context(
                closing(runtime.ndarray(dtype, size, element_shape=shape))
            )
            for name, dtype, size, shape in (
                ("input", "f32", 17, (3,)),
                ("output", "f32", 17, (3,)),
                ("keys", "u64", 17, ()),
                ("guard", "i32", 1, ()),
            )
        }
        call = stack.enter_context(closing(loaded.prepare_execution("sort", arrays)))
        for shift in (0, 3):
            keys = (np.arange(17, dtype=np.uint64) * 7 + shift) % 5 + np.uint64(1 << 60)
            bits = np.arange(51, dtype=np.uint32) + np.uint32(0x3F800000)
            bits[:4] = [0x80000000, 0x7FC00017, 0xFFC00023, 0x7F800000]
            payload = bits.reshape(17, 3)
            arrays["keys"].write(keys.tobytes())
            arrays["input"].write(payload.tobytes())
            arrays["guard"].write(np.asarray([1234567], np.int32).tobytes())
            call()
            assert (
                arrays["output"].read()
                == payload[np.argsort(keys, kind="stable")].tobytes()
            )
            assert arrays["guard"].read() == np.asarray([1234567], np.int32).tobytes()
    tree.destroy()
