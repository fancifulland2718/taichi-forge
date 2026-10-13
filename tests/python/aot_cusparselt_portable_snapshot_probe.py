"""Source-free consumers never receive or re-upload the sparse A weights."""

import sys

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_cusparselt_probe import read, write, half
from tests.python.aot_cusparselt_recording_probe import M, N, K, inputs
from tests.python.aot_cutensor_probe import near, raises


def check(arrays, host):
    a = inputs(1)["a"]
    for output, right, addend, alpha, grad in (("d", "b", "c", 0.75, "dc"), ("e", "b2", "c2", -0.5, "dc2")):
        values = [
            half(alpha * sum(a[i * K + k] * host[right][j * K + k] for k in range(K)) + 0.25 * host[addend][i * N + j])
            for i in range(M)
            for j in range(N)
        ]
        near(read(arrays[output]), values)
        near(read(arrays[grad], "f"), [0.25 * value for value in values])


def main(mode, artifact, adapter, vendor, workspace, retained):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        runtime.configure_provider("cusparselt", adapter_path=adapter, library_path=vendor)
        module = runtime.load_module(artifact)
        assert set(module.execution_plan("step").argument_names) == {"b", "b2", "c", "c2", "d", "e", "dc", "dc2"}

        def frame(factor):
            host = {name: [factor * v for v in values] for name, values in inputs(1).items() if name != "a"}
            arrays = {
                name: runtime.ndarray(
                    "f32" if name in ("dc", "dc2") else "f16", (N, K) if name in ("b", "b2") else (M, N)
                )
                for name in module.execution_plan("step").argument_names
            }
            for name, values in host.items():
                write(arrays[name], values)
            write(arrays["d"], [-77] * (M * N))
            call = module.prepare_execution("step", arrays)
            assert call.info["resource_bytes"] == M * K * 2
            assert call.info["workspace_bytes"] == int(workspace)
            assert read(arrays["d"]) == [-77] * (M * N)
            return arrays, host, call

        left, host, one = frame(1)
        right, other, two = frame(-0.5)
        one.initialize()
        assert one.info["resource_bytes"] == 0
        assert one.info["workspace_bytes"] == int(retained)
        assert read(left["d"]) == [-77] * (M * N), "Initialization executed replay"
        for _ in range(3):
            one()
            two()
        check(left, host)
        check(right, other)
        assert two.info["resource_bytes"] == 0
        one.initialize()  # Does not reconstruct or compress the fixed epoch again.
        host["b"] = [v * 0.5 for v in host["b"]]
        write(left["b"], host["b"])
        one()
        check(left, host)
        check(right, other)
        with module.prepare_execution("again", left) as replacement:
            replacement()
            check(left, host)
        one.close()
        two()
        if mode == "borrowed":
            ti.reset()
        else:
            module.close()
        raises(two)
        two.close()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
