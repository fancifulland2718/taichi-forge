"""Native lexicographic sort retains its sort/gather/affine execution closure."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import taichi_forge as ti
from tests import test_utils


def _graph(sequence):
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    return builder.compile()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_multikey_sort_export_runtime_only_and_borrowed(tmp_path, monkeypatch):
    arch = (
        "cpu"
        if ti.cfg.arch == ti.cpu
        else "cuda" if ti.cfg.arch == ti.cuda else "vulkan"
    )
    method = "cpu_native" if arch == "cpu" else "auto"
    module = ti.aot.Module()
    cases = []

    @ti.kernel
    def seed(
        dx: ti.types.ndarray(ti.f32, 1),
        loss: ti.types.ndarray(ti.f32, 1),
        dl: ti.types.ndarray(ti.f32, 1),
    ):
        for i in dx:
            dx[i] = 0
        loss[0], dl[0] = 0, 1

    @ti.kernel
    def energy(
        order: ti.types.ndarray(ti.i32, 1),
        x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        loss: ti.types.ndarray(ti.f32, 1, needs_grad=True),
    ):
        for i in order:
            j = order[i]
            loss[0] += (i + 1) * x[j] * x[j]

    for index, (dtypes, payload) in enumerate(
        (((ti.i32, ti.f32, ti.u64), True), ((ti.f64, ti.i64, ti.u32), False))
    ):
        n = 17
        owners = {f"key{k}": ti.ndarray(dtype, n) for k, dtype in enumerate(dtypes)}
        if payload:
            owners["order"] = ti.ndarray(ti.i32, n)
        sequence = ti.algorithms.primitive_sequence().sort_by_key(
            list(owners.values())[: len(dtypes)], owners.get("order"), method=method
        )
        graph = _graph(sequence)

        def forbid(*args, **kwargs):
            raise AssertionError(
                "Export must not execute or allocate producer sort scratch"
            )

        with monkeypatch.context() as patch:
            patch.setattr(sequence, "prewarm", forbid)
            patch.setattr(sequence._calls[0].workspace, "_get_order_buffer", forbid)
            patch.setattr(
                sequence._calls[0].workspace, "_get_scalar_temp_buffer", forbid
            )
            plan = module.add_execution_plan(
                f"cold{index}", graph, source_bindings=owners
            )
        description = json.loads(plan.to_json())
        assert sum(op["kind"] == "sort" for op in description["operations"]) == len(
            dtypes
        )
        # One order buffer and one scratch array per scalar dtype, independent
        # of how many passes use them. No producer data snapshot is necessary.
        private_arrays = [r for r in description["resources"] if r["kind"] == "ndarray"]
        assert len(private_arrays) == 1 + len(
            set(str(d) for d in dtypes) | ({"i32"} if payload else set())
        )
        samples = []
        for shift in (0, 3):
            hosts = {}
            for k, dtype in enumerate(dtypes):
                if dtype in (ti.f32, ti.f64):
                    data = [
                        math.nan,
                        -0.0,
                        0.0,
                        math.inf,
                        -math.inf,
                        2,
                        2,
                        -1,
                        math.nan,
                    ] + list(range(8))
                elif dtype in (ti.i64, ti.u64):
                    data = [(1 << 60) + ((i * 7) % 5) for i in range(n)]
                else:
                    data = [(i * 5) % 3 for i in range(n)]
                data = data[shift:] + data[:shift]
                hosts[f"key{k}"] = np.asarray(
                    data,
                    dtype={
                        ti.i32: np.int32,
                        ti.u32: np.uint32,
                        ti.i64: np.int64,
                        ti.u64: np.uint64,
                        ti.f32: np.float32,
                        ti.f64: np.float64,
                    }[dtype],
                )
            if payload:
                hosts["order"] = np.arange(n, dtype=np.int32)
            expected_order = sorted(
                range(n),
                key=lambda i: tuple(
                    (
                        math.isnan(float(hosts[f"key{k}"][i])),
                        (
                            0
                            if math.isnan(float(hosts[f"key{k}"][i]))
                            else hosts[f"key{k}"][i]
                        ),
                    )
                    for k in range(len(dtypes))
                ),
            )
            for name, data in hosts.items():
                owners[name].from_numpy(data)
            sequence.run()
            if payload:
                np.testing.assert_array_equal(
                    owners["order"].to_numpy(), expected_order
                )
            for k in range(len(dtypes)):
                np.testing.assert_array_equal(
                    owners[f"key{k}"].to_numpy(), hosts[f"key{k}"][expected_order]
                )
            samples.append(
                dict(
                    inputs={name: data.tobytes().hex() for name, data in hosts.items()},
                    expected={
                        name: owner.to_numpy().tobytes().hex()
                        for name, owner in owners.items()
                    },
                )
            )
        before = {name: owner.to_numpy().tobytes() for name, owner in owners.items()}
        module.add_execution_plan(f"warm{index}", graph, source_bindings=owners)
        assert before == {
            name: owner.to_numpy().tobytes() for name, owner in owners.items()
        }
        if payload:
            builder = ti.graph.GraphBuilder()
            builder.append_native(sequence)
            symbol = lambda name, dtype=ti.f32: ti.graph.Arg(
                ti.graph.ArgKind.NDARRAY, name, dtype, ndim=1
            )
            x, dx, loss, dl = (symbol(name) for name in ("x", "dx", "loss", "dl"))
            builder.dispatch(seed, dx, loss, dl)
            builder.dispatch(energy, symbol("order", ti.i32), x, loss)
            builder.dispatch(energy.grad, symbol("order", ti.i32), x, loss)
            gradient_graph = builder.compile()
            module.add_execution_plan(
                "gradient",
                gradient_graph,
                source_bindings=owners,
                gradients=dict(x="dx", loss="dl"),
            )
            gradient_graph.close()
        cases.append(
            dict(
                index=index,
                n=n,
                dtypes={name: str(owner.dtype) for name, owner in owners.items()},
                samples=samples,
            )
        )
        graph.close()
        sequence.clear()
    assert len(module._exported_kernels) == 4
    artifact = tmp_path / "multikey.tcm"
    module.archive(str(artifact))
    artifact.with_suffix(".json").write_text(json.dumps(cases), encoding="utf-8")
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_multikey_sort_probe.py")),
                mode,
                arch,
                str(artifact),
            ],
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_multikey_sort_preflight_and_small_inputs(tmp_path):
    keys = [ti.ndarray(ti.i32, 3), ti.ndarray(ti.i32, 3)]
    graph = _graph(ti.algorithms.primitive_sequence().sort_by_key(keys))
    module = ti.aot.Module()
    with pytest.raises(ti.TaichiRuntimeError, match="CPU auto host strategy"):
        module.add_execution_plan(
            "bad", graph, source_bindings=dict(a=keys[0], b=keys[1])
        )
    assert not module._content and not module._exported_kernels
    graph.close()
    graph = _graph(
        ti.algorithms.primitive_sequence().sort_by_key(
            [keys[0], keys[0]], method="cpu_native"
        )
    )
    with pytest.raises(ti.TaichiRuntimeError, match="disjoint"):
        module.add_execution_plan("bad", graph, source_bindings=dict(a=keys[0]))
    graph.close()
    for n in (1,):
        owners = {name: ti.ndarray(ti.i32, n) for name in ("a", "b")}
        graph = _graph(
            ti.algorithms.primitive_sequence().sort_by_key(
                list(owners.values()), method="cpu_native"
            )
        )
        plan = module.add_execution_plan(f"size{n}", graph, source_bindings=owners)
        assert all(
            op["kind"] == "copy" and op["size_bytes"] == 0
            for op in json.loads(plan.to_json())["operations"]
        )
        graph.close()
    artifact = tmp_path / "singleton.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded:
        owners = {name: runtime.ndarray("i32", 1) for name in ("a", "b")}
        for owner in owners.values():
            owner.write(np.array([37], dtype=np.int32).tobytes())
        with loaded.prepare_execution("size1", owners) as call:
            call()
            assert all(
                np.frombuffer(owner.read(), dtype=np.int32).tolist() == [37]
                for owner in owners.values()
            )
        for owner in owners.values():
            owner.close()
