"""Compact private storage rings retain real kernels and borrowed owners."""

import json
import struct
import time
from zipfile import ZipFile

import pytest
import taichi_forge as ti

from taichi_forge_runtime.aot import Runtime
from tests import test_utils
from tests.python.aot_graph_recording_probe import read, write


@pytest.mark.parametrize("ring", [False, True], ids=["plain-reference", "storage-ring"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_aot_storage_ring_primal_ad_and_owners(tmp_path, ring):
    arch = str(ti.lang.impl.current_cfg().arch).split(".")[-1]
    if arch in ("x64", "arm64"):
        arch = "cpu"

    @ti.kernel
    def advance(state: ti.types.ndarray(ti.f32, ndim=1)):
        for i in state:
            state[i] += 1

    @ti.kernel
    def square(
        x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
        y: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
    ):
        for i in x:
            y[i] = x[i] * x[i]

    module = ti.aot.Module()
    module.add_kernel(advance)
    module.add_kernel(square)
    module.add_kernel(square.grad, name="reverse")
    measurements = []
    configurations = ((1, 5), (3, 5), (65, 1)) if ring else ((1, 5),)
    for capacity, size in configurations:
        recipe = dict(
            kind="ordered_temporary_ring",
            version=1,
            capacity=capacity,
            arrays=["state"],
        )
        state = dict(
            kind="ndarray",
            name="state",
            dtype="f32",
            shape=[size],
            zero_initialize=True,
        )
        if capacity in (1, 3):
            state.pop("zero_initialize")
            state["initial_data"] = module.add_data(struct.pack("<f", 7.0) * size)
        resources = [state]
        # The existing selected pool provider is CUDA-only. CPU/Vulkan retain
        # their ordinary allocation contract, rather than inventing a provider.
        if arch == "cuda":
            state["allocator"] = "pool"
            resources.append(
                dict(
                    kind="native",
                    name="pool",
                    provider="memory",
                    operation="pool",
                    version=1,
                    parameters={},
                    bindings={},
                )
            )
        module.add_execution_plan(
            f"step_{capacity}",
            ti.aot.ExecutionPlan(
                [
                    dict(kind="fill32", destination="dx", value=0),
                    dict(kind="fill32", destination="dy", value=0x3F800000),
                    dict(kind="copy", source="state", destination="history"),
                    dict(kind="kernel", entry="advance", arguments=["state"]),
                    dict(kind="kernel", entry="square", arguments=["x", "y"]),
                    dict(kind="kernel", entry="reverse", arguments=["x", "y"]),
                ],
                resources=resources,
                gradients=dict(x="dx", y="dy"),
                storage_recipe=recipe if ring else None,
            ),
        )
    artifact = tmp_path / "storage.tcm"
    module.archive(str(artifact))
    with ZipFile(artifact) as archive:
        code = {
            entry: archive.read(entry)
            for entry in archive.namelist()
            if entry.endswith((".bc", ".spv", ".ptx", ".cubin"))
        }
        assert code
    for mode, factory in (
        ("standalone", lambda: Runtime(arch)),
        ("borrowed", ti.aot.borrow_runtime),
    ):
        with factory() as runtime:
            for capacity, size in configurations:
                base = 7.0 if capacity in (1, 3) else 0.0
                one = runtime.load_module(artifact)
                two = runtime.load_module(artifact)
                frames = []
                for loaded, shift in ((one, -2), (two, 3)):
                    arrays = {
                        name: runtime.ndarray("f32", size)
                        for name in ("x", "y", "dx", "dy", "history")
                    }
                    values = [float(i + shift) for i in range(size)]
                    write(arrays["x"], values)
                    write(arrays["history"], [-77.0] * size)
                    began = time.perf_counter_ns()
                    frame = loaded.prepare_execution(f"step_{capacity}", arrays)
                    measurements.append(
                        dict(
                            mode=mode,
                            capacity=capacity,
                            size=size,
                            prepare_ms=(time.perf_counter_ns() - began) / 1e6,
                            info=frame.info,
                        )
                    )
                    assert read(arrays["history"]) == [-77.0] * size
                    assert frame.info["resource_bytes"] == capacity * size * 4
                    assert frame.info["dispatch_count"] == 3
                    frames.append((frame, arrays, values))
                first, arrays, values = frames[0]
                sibling, other, other_values = frames[1]
                # The first wrap proves that separate slots retain independent
                # state; the second queues through reuse without host polling.
                for index in range(capacity + 1):
                    first()
                    assert (
                        read(arrays["history"])
                        == [base + float(index // capacity)] * size
                    )
                for _ in range(capacity):
                    first()
                assert read(arrays["history"]) == [base + 2.0] * size
                assert read(arrays["y"]) == [x * x for x in values]
                assert read(arrays["dx"]) == [2 * x for x in values]
                sibling()
                assert read(other["history"]) == [base] * size
                assert read(other["dx"]) == [2 * x for x in other_values]
                samples = []
                for _ in range(3):
                    began = time.perf_counter_ns()
                    first()
                    submitted = time.perf_counter_ns()
                    runtime.wait()
                    completed = time.perf_counter_ns()
                    samples.append(
                        dict(
                            host_us=(submitted - began) / 1000,
                            complete_us=(completed - began) / 1000,
                        )
                    )
                measurements[-2]["samples"] = samples
                invalid = runtime.ndarray("i32", size)
                before_failure = read(arrays["history"])
                with pytest.raises(RuntimeError):
                    one.prepare_execution(f"step_{capacity}", dict(arrays, x=invalid))
                assert read(arrays["history"]) == before_failure
                invalid.close()
                # Reprepare starts fresh storage, while existing state survives.
                fresh = one.prepare_execution(f"step_{capacity}", arrays)
                fresh()
                assert read(arrays["history"]) == [base] * size
                fresh.close()
                first()
                arrays["x"].close()
                with pytest.raises(RuntimeError):
                    first()
                first.close()
                one.close()
                sibling()
                two.close()  # Retires submitted work before module storage.
                assert read(other["history"]) == [base + 1 // capacity] * size
                with pytest.raises(RuntimeError):
                    sibling()
                sibling.close()
                sibling.close()
                for bound in (arrays, other):
                    for value in bound.values():
                        value.close()
    # The borrowed Program must retire every frame before its device disappears.
    runtime = ti.aot.borrow_runtime()
    loaded = runtime.load_module(artifact)
    arrays = {
        name: runtime.ndarray("f32", 5) for name in ("x", "y", "dx", "dy", "history")
    }
    write(arrays["x"], [1, 2, 3, 4, 5])
    pending = loaded.prepare_execution("step_3" if ring else "step_1", arrays)
    pending()
    ti.reset()
    with pytest.raises(RuntimeError):
        pending()
    for value in arrays.values():
        value.close()
    pending.close()
    loaded.close()
    runtime.close()
    runtime.close()
    (tmp_path / "artifact.json").write_text(
        json.dumps(
            dict(
                arch=arch,
                ring=ring,
                measurements=measurements,
                artifact_bytes=artifact.stat().st_size,
                code_entries=len(code),
                code_bytes=sum(map(len, code.values())),
            )
        ),
        encoding="utf-8",
    )
