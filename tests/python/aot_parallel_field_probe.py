"""Consume automatically lowered parallel Field snapshots without the DSL."""

import array
import os
import sys


class NoFrontend:
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in ("taichi", "taichi_forge", "numpy"):
            raise AssertionError("Source-free consumer imported " + fullname)


sys.meta_path.insert(0, NoFrontend())
from taichi_forge_runtime.aot import Runtime, Scalar

backend, artifact = sys.argv[1:]
options = dict(device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0"))) if backend == "vulkan" else {}
with Runtime(backend, **options) as runtime:
    module = runtime.load_module(artifact)
    domain = runtime.execution_domain(capacity=2, instance_capacity=2)
    calls = [module.prepare_execution("step", dict(slot=Scalar("i32", slot), epoch=Scalar("u64", 0)),
                                     execution_domain=domain, mutable_parameters=("epoch",)) for slot in (0, 1)]
    for call in calls:
        call.initialize()
    out = runtime.ndarray("f32", 3 * 17 * 3)
    header = runtime.ndarray("u64", 4)
    for slot in (0, 1):
        done = calls[slot].submit(values=dict(epoch=(1 << 63) + slot))
        assert done is not None
        done.wait()
        module.kernel("readback").bind(out, header, Scalar("i32", slot)).run()
        values, metadata = array.array("f"), array.array("Q")
        values.frombytes(out.read())
        metadata.frombytes(header.read())
        assert values.tolist() == [float(lane * 100 + i) for lane in range(3) for i in range(51)]
        assert metadata.tolist() == [(1 << 63) + slot, *[slot + 10] * 3]
        done.close()
    for call in calls:
        call.close()
    domain.close()
    module.close()
print("PARALLEL_FIELD_OK")
