"""No DSL in standalone mode; explicit per-branch AD, not AD of control flow."""

import json
import struct
import sys
import time

from taichi_forge_runtime import aot


def frame(runtime, module, name, scale=1, timings=None):
    begin = time.perf_counter()
    arrays = {n: runtime.ndarray("f32", 17) for n in ("x", "y", "gradient")}
    arrays["choice"] = runtime.ndarray("i32", 1)
    arrays["state"] = runtime.ndarray("i32", 2)
    arrays["x"].write(struct.pack("<17f", *[scale * (i - 8) for i in range(17)]))
    arrays["y"].write(struct.pack("<17f", *([71] * 17)))
    arrays["choice"].write(struct.pack("<i", 0))
    bindings_ready = time.perf_counter()
    call = module.prepare_execution(name, arrays)
    if timings is not None:
        timings.update(
            binding_ms=(bindings_ready - begin) * 1000, prepare_ms=(time.perf_counter() - bindings_ready) * 1000
        )
    assert struct.unpack("<17f", arrays["y"].read()) == (71,) * 17, "Cold prepare executed a body"
    return arrays, call


def expect(arrays, name, value, scale=1):
    fallback = name in ("if_else", "switch_default")
    checks, repeats = 1, 1
    if name == "nested":
        checks, repeats = (3, 2) if value else (1, 0)
        gain = value + 1 if value and 0 <= value < 8 else 0
    elif name.startswith("if"):
        gain = 1 if value else 2 if fallback else 0
    else:
        gain = value + 1 if 0 <= value < 8 else 9 if fallback else 0
    if name == "switch_empty" and value == 7:
        gain = 0
    assert struct.unpack("<2i", arrays["state"].read()) == (checks, -77 if gain else value)
    xs = [scale * (i - 8) for i in range(17)]
    assert list(struct.unpack("<17f", arrays["y"].read())) == ([gain * x * x for x in xs] if gain else [-91] * 17)
    gradient = list(struct.unpack("<17f", arrays["gradient"].read()))
    assert gradient == [repeats * 2 * gain * x for x in xs], (name, value, gain, gradient)


def check(runtime, arch, path, host=False, masked=False):
    start = time.perf_counter()
    module = runtime.load_module(path)
    loaded = time.perf_counter()
    stats = dict(load_ms=(loaded - start) * 1000)
    for name in ("if", "if_else", "switch", "switch_default", "switch_empty" if masked else "nested"):
        costs = {}
        arrays, call = frame(runtime, module, name, timings=costs)
        arrays["choice"].write(struct.pack("<i", 2))
        first = time.perf_counter()
        call()
        runtime.wait()
        costs["first_complete_ms"] = (time.perf_counter() - first) * 1000
        expect(arrays, name, 2)
        for value in (-2147483648, -1, 0, 1, 2, 7, 8, 2147483647):
            arrays["choice"].write(struct.pack("<i", value))
            call()
            expect(arrays, name, value)
        info = call.info
        assert info["launches"] == 9
        expected_dispatches = (
            {"if": 4, "if_else": 7, "switch": 25, "switch_default": 28, "switch_empty": 22}
            if masked
            else {"if": 3, "if_else": 5, "switch": 17, "switch_default": 19, "nested": 35 if host else 18}
        )
        assert info["dispatch_count"] == expected_dispatches[name]
        if host:
            assert info["execution_kind"] == "host_control"
            assert info["workspace_bytes"] == (4 if arch == "vulkan" else 0)
        elif arch == "cuda":
            assert info["execution_kind"] == "cuda_graph"
            assert info["workspace_bytes"] == (4 if masked else 80 if name == "nested" else 24)
        # Excludes readback of correctness results; completion is one owner wait.
        arrays["choice"].write(struct.pack("<i", 2))
        begin_warm = time.perf_counter()
        for _ in range(32):
            call()
        submitted = time.perf_counter()
        runtime.wait()
        completed = time.perf_counter()
        expect(arrays, name, 2)
        stats[name] = dict(
            **costs,
            host_us=(submitted - begin_warm) * 1e6 / 32,
            complete_us=(completed - begin_warm) * 1e6 / 32,
            workspace_bytes=info["workspace_bytes"],
            argument_bytes=info["argument_bytes"],
            dispatch_count=info["dispatch_count"],
        )
        call.close()
        for array in arrays.values():
            array.close()

    if arch == "cuda" and not masked:
        stats["provider"] = provider_branch(runtime, module, host=host)

    first, one = frame(runtime, module, "switch")
    second, two = frame(runtime, module, "switch_default", scale=3)
    first["choice"].write(struct.pack("<i", 2))
    second["choice"].write(struct.pack("<i", -1))
    one()
    two()
    expect(first, "switch", 2)
    expect(second, "switch_default", -1, 3)

    # Invalid selector and route fail before any work; another preparation recovers.
    region = dict(
        kind="branch",
        selector="selector",
        selection="nonzero",
        has_default=False,
        condition=[],
        bodies=[[]],
        lowering=dict(
            kind=(
                "cuda_masked" if masked else "cpu" if arch == "cpu" else f"{arch}_host" if host else "cuda_conditional"
            ),
            version=1,
        ),
    )
    for dtype, size in (("f32", 1), ("i32", 2)):
        invalid = runtime.ndarray(dtype, size)
        try:
            module.prepare_execution(aot.ExecutionPlan([region]), dict(selector=invalid))
        except aot.AotError as error:
            assert "one aligned scalar i32" in str(error), str(error)
        else:
            raise AssertionError("Invalid selector was accepted")
        invalid.close()
    other = dict(region, lowering=dict(kind="cuda_conditional" if arch == "cpu" else "cpu", version=1))
    try:
        module.prepare_execution(aot.ExecutionPlan([other]), dict(selector=first["choice"]))
    except aot.AotError as error:
        assert "declared backend lowering" in str(error), str(error)
    else:
        raise AssertionError("Backend lowering silently changed")
    empty = module.prepare_execution(aot.ExecutionPlan([region]), dict(selector=first["choice"]))
    empty()
    if masked:
        assert empty.info["workspace_bytes"] == empty.info["argument_bytes"] == 0
    empty.close()
    if masked:
        bad = dict(region, bodies=[[dict(kind="kernel", entry="forward", arguments=["x", "y", "gain"])]])
        try:
            module.prepare_execution(
                aot.ExecutionPlan([bad], gradients={"x": "gradient", "y": "dy"}),
                dict(
                    selector=first["choice"],
                    x=first["x"],
                    y=first["y"],
                    gradient=first["gradient"],
                    dy=second["gradient"],
                    gain=aot.Scalar("f32", 1),
                ),
            )
        except aot.AotError as error:
            assert "Unsupported fixed CUDA Graph dispatch" in str(error), str(error)
        else:
            raise AssertionError("Ordinary body entry was accepted as gated code")
    one()
    first["x"].close()  # Closing a dependency retires submitted work.
    try:
        one()
    except (RuntimeError, aot.AotError):
        pass
    else:
        raise AssertionError("Closed branch dependency was reused")
    one.close()
    two()
    expect(second, "switch_default", -1, 3)
    module.close()
    two.close()
    for arrays in (first, second):
        for array in arrays.values():
            array.close()
    return stats


def provider_branch(runtime, module, host=False):
    available = runtime.probe_native_operation("gemm_f32")["available"]
    if not available:
        return "cuBLAS unavailable; provider branch not exercised"
    arrays = {name: runtime.ndarray("f32", (2, 2)) for name in ("a", "b", "out", "adjoint")}
    arrays["selector"] = runtime.ndarray("i32", 1)
    arrays["a"].write(struct.pack("<4f", 1, 2, 3, 4))
    arrays["b"].write(struct.pack("<4f", 2, 0, 0, 3))
    arrays["out"].write(struct.pack("<4f", *([71] * 4)))

    def gemm(a, b, out, **parameters):
        return dict(
            kind="native",
            provider="cublas",
            operation="gemm_f32",
            version=1,
            parameters=dict(rows=2, columns=2, inner=2, **parameters),
            bindings=dict(a=a, b=b, output=out),
        )

    branch = dict(
        kind="branch",
        selector="selector",
        selection="nonzero",
        has_default=False,
        lowering=dict(kind="cuda_host" if host else "cuda_conditional", version=1),
        condition=[],
        bodies=[[gemm("a", "b", "out"), gemm("out", "b", "adjoint", transpose_b=True)]],
    )
    plan = aot.ExecutionPlan([dict(kind="fill32", destination=name, value=0) for name in ("out", "adjoint")] + [branch])
    call = module.prepare_execution(plan, arrays)
    assert struct.unpack("<4f", arrays["out"].read()) == (71,) * 4
    for selected in (0, 1, 0, -1):
        arrays["selector"].write(struct.pack("<i", selected))
        call()
        assert struct.unpack("<4f", arrays["out"].read()) == ((2, 6, 6, 12) if selected else (0,) * 4)
        # Explicit adjoint of 0.5*sum((A B)^2): dA = (A B) B^T.
        assert struct.unpack("<4f", arrays["adjoint"].read()) == ((4, 18, 12, 36) if selected else (0,) * 4)
    call()
    call.close()
    for array in arrays.values():
        array.close()
    return "cuBLAS branch and explicit matrix adjoint passed"


if __name__ == "__main__":
    mode, arch, path = sys.argv[1:4]
    host = sys.argv[4:] == ["host"]
    masked = sys.argv[4:] == ["masked"]
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=getattr(ti, arch), offline_cache=False)
        with ti.aot.borrow_runtime() as runtime:
            stats = check(runtime, arch, path, host=host, masked=masked)
            module = runtime.load_module(path)
            arrays, call = frame(runtime, module, "if_else")
            call()
            ti.reset()
            try:
                call()
            except (RuntimeError, aot.AotError):
                pass
            else:
                raise AssertionError("Owner reset did not retire branches")
            call.close()
    else:
        with aot.Runtime(arch) as runtime:
            stats = check(runtime, arch, path, host=host, masked=masked)
        assert "taichi_forge" not in sys.modules
    print("RESULT " + json.dumps(stats, sort_keys=True))
