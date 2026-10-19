"""Cold/warm prepared tensor recordings reuse the AOT storage/export contract."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=[ti.cuda, ti.vulkan], offline_cache=False)
def test_prepared_tensor_sort_aot_views_fields_and_downstream_grad(
    tmp_path, monkeypatch
):
    module = ti.aot.Module()
    cases = []
    trees = []
    for index, dtype, npdtype, shape, storage in (
        (0, ti.f32, np.float32, (3,), "view"),
        (1, ti.i64, np.int64, (2, 2), "ndarray"),
        (2, ti.f32, np.float32, (3,), "field"),
        (4, ti.i32, np.int32, (3,), "ndarray"),
    ):
        n = 19 if storage == "view" else 17
        keys = ti.ndarray(ti.u64, n)
        if storage == "field":
            builder = ti.FieldsBuilder()
            pad, values = ti.field(ti.i32), ti.Vector.field(3, dtype)
            builder.dense(ti.i, 1).place(pad)
            builder.dense(ti.i, n).place(values)
            tree = builder.finalize()
            trees.append(tree)
            pad[0] = 1234567
        else:
            values = (
                ti.Vector.ndarray(shape[0], dtype, n)
                if len(shape) == 1
                else ti.Matrix.ndarray(*shape, dtype, n)
            )
        selection = slice(1, n - 1) if storage == "view" else slice(None)
        owners = [keys, values]
        if storage == "view":
            owners = [
                ti.experimental.ndarray_view(a, slices=(selection,)) for a in owners
            ]
        prepared = ti.algorithms.prepare_sort(*owners)
        builder = ti.graph.GraphBuilder()
        builder.append_native(prepared.record())
        graph = builder.compile()
        bindings = dict(keys=keys)
        if storage != "field":
            bindings["values"] = values

        def export(phase):
            def forbidden(*args, **kwargs):
                raise AssertionError("Export must not execute prepared sorting")

            with monkeypatch.context() as patch:
                patch.setattr(prepared, "run", forbidden)
                plan = module.add_execution_plan(
                    f"source_{phase}{index}",
                    graph,
                    source_bindings=bindings,
                    snapshot_state=storage == "field",
                )
            document = json.loads(plan.to_json())
            assert [op["kind"] for op in document["operations"]] == ["sort"]
            if storage == "field":
                # Exercise the automatically reconstructed private Field via
                # public IO; copies preserve raw bits without a helper kernel.
                private = document["operations"][0]["values"]
                document["operations"] = [
                    dict(kind="copy", source="values", destination=private),
                    *document["operations"],
                    dict(kind="copy", source=private, destination="values"),
                ]
            module.add_execution_plan(
                f"{phase}{index}", ti.aot.ExecutionPlan.from_json(json.dumps(document))
            )
            assert not module._exported_kernels
            return document

        samples = []
        for shift in (0, 3):
            source = (np.arange(n, dtype=np.uint64) * 7 + shift) % 5 + np.uint64(
                1 << 60
            )
            payload = (
                np.arange(n * np.prod(shape), dtype=npdtype).reshape((n, *shape))
                + shift
            )
            if dtype == ti.f32:
                payload.view(np.uint32).reshape(-1)[:5] = [
                    0x80000000,
                    0,
                    0x7FC00017,
                    0xFFC00023,
                    0x7F800000,
                ]
            elif dtype == ti.i64:
                payload += np.int64(1 << 60)
            if index == 4:
                payload[:, 0] = np.arange(n)
            keys.from_numpy(source)
            values.from_numpy(payload)
            if shift == 0:
                initial_payload = payload.copy()
                document = export("cold")
                assert values.to_numpy().tobytes() == payload.tobytes()
            prepared.run()
            order = np.arange(n)
            order[selection] = order[selection][
                np.argsort(source[selection], kind="stable")
            ]
            assert keys.to_numpy().tobytes() == source[order].tobytes()
            assert values.to_numpy().tobytes() == payload[order].tobytes()
            samples.append(
                dict(
                    keys=source.tobytes().hex(),
                    values=payload.tobytes().hex(),
                    sorted_keys=source[order].tobytes().hex(),
                    sorted_values=payload[order].tobytes().hex(),
                    order=order.tolist(),
                )
            )
        if storage == "field":
            # One module owns one coherent initial root snapshot, independent
            # of whether the producer's prepared backend has executed before.
            values.from_numpy(initial_payload)
        export("warm")
        if index == 4:
            gradient_document = document
        cases.append(
            dict(
                index=index,
                key_dtype="u64",
                dtype=str(dtype),
                shape=shape,
                n=n,
                entries=[f"cold{index}", f"warm{index}"],
                samples=samples,
            )
        )
        graph.close()
        prepared.close()
        if storage == "field":
            assert pad[0] == 1234567

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

    module.add_kernel(seed)
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="backward")
    gradient_document["operations"] += [
        dict(kind="kernel", entry="seed", arguments=["dx", "loss", "dl"]),
        dict(kind="kernel", entry="energy", arguments=["values", "x", "loss"]),
        dict(kind="kernel", entry="backward", arguments=["values", "x", "loss"]),
    ]
    gradient_document["gradients"] = dict(x="dx", loss="dl")
    module.add_execution_plan(
        "gradient", ti.aot.ExecutionPlan.from_json(json.dumps(gradient_document))
    )
    artifact = tmp_path / "prepared-tensor-sort.tcm"
    module.archive(str(artifact))
    for tree in trees:
        tree.destroy()
    artifact.with_suffix(".json").write_text(json.dumps(cases), encoding="utf-8")
    arch = "cuda" if ti.cfg.arch == ti.cuda else "vulkan"
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [
                sys.executable,
                *(["-S"] if mode == "standalone" else []),
                str(Path(__file__).with_name("aot_tensor_sort_probe.py")),
                mode,
                arch,
                str(artifact),
            ],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
