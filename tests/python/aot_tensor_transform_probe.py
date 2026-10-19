"""Tensor transform consumers use raw buffers and normal installed runtime APIs."""

import array
from contextlib import ExitStack, closing
import json
import math
import os
from pathlib import Path
import sys

from taichi_forge_runtime.aot import ExecutionPlan, Runtime, Scalar
from aot_transform_probe import check
from aot_scan_probe import CODES, reject


def main(mode, arch, artifact, kind):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False, cpu_max_num_threads=4)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime(arch, device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")) if arch == "vulkan" else 0)
    with runtime, runtime.load_module(artifact) as module:
        if kind == "gradient":
            frame = {name: runtime.ndarray("f32", 7, element_shape=(3,)) for name in ("src", "dst", "x", "dx")}
            frame.update({name: runtime.ndarray("f32", 1) for name in ("loss", "dl")})
            with module.prepare_execution("gradient", frame) as call:
                for iteration in (1, 0, 2):
                    source = [float(i * iteration) for i in range(21)]
                    x = [float(i % 5) for i in range(21)]
                    frame["src"].write(array.array("f", source))
                    frame["x"].write(array.array("f", x))
                    call()
                    check(frame["dx"].read(), array.array("f", [2*b*(3*a+7) for a,b in zip(source,x)]).tobytes(), "f32")
                    check(frame["loss"].read(), array.array("f", [sum((3*a+7)*b*b for a,b in zip(source,x))]).tobytes(), "f32")
            for value in frame.values():
                value.close()
        else:
            cases = json.loads(Path(artifact).with_suffix(".json").read_text())
            for case in cases:
                with ExitStack() as stack:
                    dtype, size, shape = case["dtype"], case["size"], case["shape"]
                    src = stack.enter_context(closing(runtime.ndarray(dtype, size, element_shape=shape)))
                    dst = src if case["inplace"] else stack.enter_context(closing(runtime.ndarray(dtype, size, element_shape=shape)))
                    args = dict(src=src) if case["inplace"] else dict(src=src, dst=dst)
                    for name in (case["name"], case["name"] + "_warm"):
                        with module.prepare_execution(name, args) as call:
                            assert call.info["workspace_bytes"] == 0
                            for sample in case["samples"]:
                                source = bytes.fromhex(sample["input"])
                                src.write(source)
                                call()
                                check(dst.read(), bytes.fromhex(sample["expected"]), dtype)
                                if not case["inplace"]:
                                    assert src.read() == source
                        reject(call)
            case = next(c for c in cases if c["dtype"] == "f32" and c["size"] < 1000)
            size, shape = case["size"], case["shape"]
            with closing(runtime.ndarray("f32", size+2, element_shape=shape)) as parent:
                plan = module.execution_plan(case["name"])
                views = [dict(kind="ndarray_view", name=name, array="parent", offset=offset, shape=[size])
                         for name,offset in (("src",0),("dst",1))]
                reject(lambda: module.prepare_execution(plan.with_argument_resources(views), dict(parent=parent)))
                views[0]["offset"] = 1
                sample = case["samples"][0]
                guard = array.array("f", [79]*math.prod(shape)).tobytes()
                parent.write(guard + bytes.fromhex(sample["input"]) + guard)
                with module.prepare_execution(plan.with_argument_resources(views), dict(parent=parent)) as call:
                    call()
                    actual = parent.read()
                    assert actual[:len(guard)] == guard and actual[-len(guard):] == guard
                    check(actual[len(guard):-len(guard)], bytes.fromhex(sample["expected"]), "f32")
            frames = []
            for _ in range(2):
                args = {name:runtime.ndarray("f32", size, element_shape=shape) for name in ("src","dst")}
                args["src"].write(bytes.fromhex(case["samples"][0]["input"]))
                frames.append((args, module.prepare_execution(case["name"], args)))
            # A native prefix may use a larger packed allocation; preserve all
            # tail lanes rather than flattening the prefix count incorrectly.
            operation = dict(kind="native", provider="builtin", operation="transform", version=1,
                             parameters=dict(size=2, dtype="f32", element_shape=shape,
                                             scale_bits=Scalar("f64",3).resource("s")["bits"],
                                             bias_bits=Scalar("f64",7).resource("b")["bits"]),
                             bindings=dict(src="src",dst="dst"))
            args = frames[0][0]
            sentinel = array.array("f", [79]*(size*math.prod(shape))).tobytes()
            args["dst"].write(sentinel)
            with module.prepare_execution(ExecutionPlan([operation]), args) as call:
                call()
                prefix_bytes = 2*math.prod(shape)*4
                expected = bytes.fromhex(case["samples"][0]["expected"])[:prefix_bytes] + sentinel[prefix_bytes:]
                check(args["dst"].read(), expected, "f32")
            for dtype,n,s in (("u32",size,shape),("f32",size-1,shape),("f32",size,[1,3]),("f32",size,[])):
                with closing(runtime.ndarray(dtype,n,element_shape=s)) as wrong:
                    reject(lambda: module.prepare_execution(case["name"], dict(src=wrong,dst=frames[0][0]["dst"])))
            frames[0][1]()
            frames[0][1].close()
            frames[1][1]()
            check(frames[1][0]["dst"].read(), bytes.fromhex(case["samples"][0]["expected"]), "f32")
            if mode == "borrowed":
                ti.reset()
            else:
                module.close()
            reject(frames[1][1])
            for args,call in frames:
                call.close()
                for value in args.values():
                    value.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules and "numpy" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
