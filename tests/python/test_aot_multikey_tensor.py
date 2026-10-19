"""Packed tensor payloads retain native multi-key sorting in cold AOT export."""

import json
import os
from pathlib import Path
import subprocess
import sys

import taichi_forge as ti
from tests import test_utils
from tests.python.test_sort_multikey_tensor import tensor_array, tensor_case


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_multikey_tensor_aot_runtime_only_and_borrowed(tmp_path, monkeypatch):
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
        order: ti.types.ndarray(ti.types.vector(3, ti.i32), 1),
        x: ti.types.ndarray(ti.f32, 1, needs_grad=True),
        loss: ti.types.ndarray(ti.f32, 1, needs_grad=True),
    ):
        for i in order:
            j = order[i][0]
            loss[0] += (i + 1) * x[j] * x[j]

    for index, (dtype, shape) in enumerate(
        ((ti.f32, (3,)), (ti.i64, (2, 2)), (ti.i32, (3,)))
    ):
        n = 17
        payload = "order" if dtype == ti.i32 else "values"
        keys = [ti.ndarray(ti.i32, n), ti.ndarray(ti.i32, n)]
        values = tensor_array(dtype, shape, n)
        owners = dict(key0=keys[0], key1=keys[1], **{payload: values})
        sequence = ti.algorithms.primitive_sequence().sort_by_key(
            keys, values, method=method
        )
        builder = ti.graph.GraphBuilder()
        builder.append_native(sequence)
        graph = builder.compile()

        def forbid(*args, **kwargs):
            raise AssertionError(
                "Cold export must not allocate or run producer reorder scratch"
            )

        with monkeypatch.context() as patch:
            patch.setattr(sequence, "prewarm", forbid)
            workspace = sequence._calls[0].workspace
            for name in (
                "_get_order_buffer",
                "_get_scalar_temp_buffer",
                "_get_order_apply_tensor_temp_buffer",
                "_get_order_apply_identity_buffer",
            ):
                patch.setattr(workspace, name, forbid)
            plan = module.add_execution_plan(
                f"cold{index}", graph, source_bindings=owners
            )
        desc = json.loads(plan.to_json())
        assert len(desc["initialization"]) == 1
        # One order, one identity, one shared scalar temporary and one tensor
        # temporary. Scalar/tensor records with the same dtype cannot alias.
        assert len([r for r in desc["resources"] if r["kind"] == "ndarray"]) == 4
        samples = []
        for shift in (0, 3):
            host_keys, host_values, order = tensor_case(dtype, shape, n, shift)
            hosts = dict(key0=host_keys[0], key1=host_keys[1], **{payload: host_values})
            for name, data in hosts.items():
                owners[name].from_numpy(data)
            sequence.run()
            for name, data in hosts.items():
                assert owners[name].to_numpy().tobytes() == data[order].tobytes()
            samples.append(
                dict(
                    inputs={k: v.tobytes().hex() for k, v in hosts.items()},
                    expected={k: v[order].tobytes().hex() for k, v in hosts.items()},
                )
            )
        before = values.to_numpy().tobytes()
        module.add_execution_plan(f"warm{index}", graph, source_bindings=owners)
        assert values.to_numpy().tobytes() == before
        if payload == "order":
            gradient_builder = ti.graph.GraphBuilder()
            gradient_builder.append_native(sequence)
            symbols = {
                name: ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, ti.f32, ndim=1)
                for name in ("x", "dx", "loss", "dl")
            }
            order_arg = ti.graph.Arg(
                ti.graph.ArgKind.NDARRAY, "order", ti.types.vector(3, ti.i32), ndim=1
            )
            gradient_builder.dispatch(
                seed, symbols["dx"], symbols["loss"], symbols["dl"]
            )
            gradient_builder.dispatch(energy, order_arg, symbols["x"], symbols["loss"])
            gradient_builder.dispatch(
                energy.grad, order_arg, symbols["x"], symbols["loss"]
            )
            gradient = gradient_builder.compile()
            module.add_execution_plan(
                "gradient",
                gradient,
                source_bindings=owners,
                gradients=dict(x="dx", loss="dl"),
            )
            gradient.close()
        cases.append(
            dict(
                index=index,
                n=n,
                dtypes={k: str(v.dtype) for k, v in owners.items()},
                shapes={payload: shape},
                samples=samples,
            )
        )
        graph.close()
        sequence.clear()
    # Identity initialization shares the existing order helper's code.
    assert len(module._exported_kernels) == 4
    artifact = tmp_path / "tensor-multikey.tcm"
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
