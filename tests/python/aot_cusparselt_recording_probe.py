"""Independent FP16 reference; differentiate a fixed sparsity pattern only."""

import sys

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_cusparselt_probe import read, write, half
from tests.python.aot_cutensor_probe import near, raises

M, N, K = 16, 32, 32


def inputs(factor, shift=0):
    return dict(
        a=[factor * (1 + (i + k) % 3) / 8 if (k + shift) % 4 < 2 else 0 for i in range(M) for k in range(K)],
        b=[((j * 3 + k) % 5 - 2) / 8 for j in range(N) for k in range(K)],
        b2=[((j + k * 2) % 7 - 3) / 16 for j in range(N) for k in range(K)],
        c=[(i % 7 - 3) / 16 for i in range(M * N)],
        c2=[(i % 9 - 4) / 16 for i in range(M * N)],
    )


def check(arrays, host, relu):
    a, b, b2 = (host[name] for name in ("a", "b", "b2"))
    alpha_e = -0.5 if relu else 0.75

    def product(right, seed, alpha):
        result = [
            alpha * sum(a[i * K + k] * right[j * K + k] for k in range(K)) + 0.25 * seed[i * N + j]
            for i in range(M)
            for j in range(N)
        ]
        return [half(max(0, value) if relu else value) for value in result]

    d, e = product(b, host["c"], 0.75), product(b2, host["c2"], alpha_e)
    da = [
        (
            sum(0.75 * d[i * N + j] * b[j * K + k] + alpha_e * e[i * N + j] * b2[j * K + k] for j in range(N))
            if a[i * K + k]
            else 0
        )
        for i in range(M)
        for k in range(K)
    ]
    db = [sum(0.75 * d[i * N + j] * a[i * K + k] for i in range(M)) for j in range(N) for k in range(K)]
    db2 = [sum(alpha_e * e[i * N + j] * a[i * K + k] for i in range(M)) for j in range(N) for k in range(K)]
    expected = dict(d=d, e=e, da=da, db=db, db2=db2, loss=[0.5 * sum(x * x + y * y for x, y in zip(d, e))])
    for name, values in expected.items():
        near(read(arrays[name], "e" if name in ("d", "e") else "f"), values)


def main(mode, artifact, adapter, vendor, case, workspace):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        runtime.configure_provider("cusparselt", adapter_path=adapter, library_path=vendor)
        module = runtime.load_module(artifact)
        shapes = dict(
            a=(M, K),
            b=(N, K),
            b2=(N, K),
            c=(M, N),
            c2=(M, N),
            d=(M, N),
            e=(M, N),
            da=(M, K),
            db=(N, K),
            db2=(N, K),
            loss=(1,),
        )

        def frame(factor):
            arrays = {
                name: runtime.ndarray("f32" if name in ("da", "db", "db2", "loss") else "f16", shape)
                for name, shape in shapes.items()
            }
            host = inputs(factor)
            for name, values in host.items():
                write(arrays[name], values)
            write(arrays["d"], [-77] * (M * N))
            run = module.prepare_execution("step", arrays)
            assert read(arrays["d"]) == [-77] * (M * N)
            assert run.info["execution_kind"] == "cuda_graph"
            assert run.info["workspace_bytes"] == int(workspace), run.info
            return arrays, host, run

        arrays, host, one = frame(1)
        other_arrays, other, two = frame(-0.5)
        for _ in range(3):
            one()
            two()
        check(arrays, host, case == "relu")
        check(other_arrays, other, case == "relu")
        host = inputs(0.5, shift=2)
        write(arrays["a"], host["a"])
        one()
        check(arrays, host, case == "relu")
        check(other_arrays, other, case == "relu")
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
