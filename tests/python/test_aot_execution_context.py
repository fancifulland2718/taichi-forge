"""Load actual AOT kernels in an independent runtime and interleave instances."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Scalar
from tests import test_utils


@test_utils.test(arch=ti.cpu, offline_cache=False, compile_tier="full")
def test_package_helpers_reuse_declared_receivers_across_instances(tmp_path, monkeypatch):
    from collections import Counter
    import gc
    import weakref
    import numpy as np
    from taichi_forge.aot._package_compilation import PackageCompilation
    from taichi_forge.lang.kernel_impl import Func
    from taichi_forge_runtime.aot import Runtime
    from tests.python.test_aot_package import _source_package

    source = '''import taichi_forge as ti
@ti.data_oriented
class Child:
    def __init__(self, bias, dtype, interleaved):
        self.bias = bias
        scope = ti.experimental.ResourceScope(self)
        scope.static('bias', choices=(3, 9))
        if interleaved:
            layout = scope.dense(ti.i, 3)
            scope.pad = scope.field(ti.i32, shape=3, placement=layout)
            scope.data = scope.field(dtype, shape=3, placement=layout)
        else:
            scope.data = scope.field(dtype, shape=3)
    @ti.func
    def leaf(self, i: ti.i32, offset: ti.template()) -> ti.i32:
        return ti.cast(self.data[i], ti.i32) + ti.static(self.bias) + offset
    @ti.func
    def parent(self, i: ti.i32, offset: ti.template()) -> ti.i32:
        return self.leaf(i, offset) * 2
class OtherChild(Child):
    pass
@ti.data_oriented
class Model:
    def __init__(self, child, other, shadow):
        self.shadow = shadow
        scope = ti.experimental.ResourceScope(self)
        scope.static('shadow', choices=(False, True))
        scope.child = child
        scope.other = other
        scope.out = scope.field(ti.i32, shape=3)
'''
    # Separate entries force a helper lookup instead of a whole-entry cache hit.
    cases = (
        ("first", "Child", "child", False, 3, 2, ti.i32, False, None, 1),
        ("with_shadow", "Child", "child", True, 3, 2, ti.i32, False, None, 1),
        ("static", "Child", "child", False, 9, 2, ti.i32, False, None, 2),
        ("template", "Child", "child", True, 9, 4, ti.i32, False, None, 3),
        ("dtype", "Child", "child", False, 9, 4, ti.f32, False, None, 4),
        ("layout", "Child", "child", False, 9, 4, ti.f32, True, None, 5),
        ("receiver_type", "OtherChild", "child", False, 9, 4, ti.f32, True, None, 6),
        ("receiver_path", "OtherChild", "other", False, 9, 4, ti.f32, True, None, 7),
        ("tier", "OtherChild", "other", False, 9, 4, ti.f32, True, "fast", 8),
    )
    for name, _, receiver, _, _, offset, _, _, tier, _ in cases:
        decorator = "@ti.kernel" if tier is None else f"@ti.kernel(opt_level={tier!r})"
        source += f'''
    {decorator}
    def {name}(self):
        for i in range(3):
            self.out[i] = self.{receiver}.parent(i, {offset})
            if ti.static(self.shadow):
                self.out[i] += 1
'''
    package = _source_package(tmp_path, monkeypatch, source)
    frontend, originals, owner_refs, snapshots = Counter(), [], [], []
    original_compile = Func.do_compile

    def observe(function, *args, **kwargs):
        if function.is_real_function:
            frontend[function.func.__name__] += 1
        return original_compile(function, *args, **kwargs)

    monkeypatch.setattr(Func, "do_compile", observe)
    with PackageCompilation() as compilation:
        module = ti.aot.Module(artifact_format="bc")
        compilation.attach(module, function_library=True)
        for name, child_type, receiver, shadow, bias, offset, dtype, interleaved, _, count in cases:
            child = getattr(package.model, child_type)(bias, dtype, interleaved)
            other = getattr(package.model, child_type)(bias, dtype, interleaved)
            values = np.arange(3, dtype=np.int32 if dtype == ti.i32 else np.float32) + 5
            child.data.from_numpy(values)
            other.data.from_numpy(values)
            owner = package.model.Model(child, other, shadow)
            owner_refs.extend((weakref.ref(owner), weakref.ref(child), weakref.ref(other)))
            expected = (values.astype(np.int32) + bias + offset) * 2 + int(shadow)
            # Ordinary JIT is the independent execution reference and remains usable.
            getattr(owner, name)()
            np.testing.assert_array_equal(owner.out.to_numpy(), expected)
            component = ti.aot.Component.from_owner(owner)
            code = component.compile(compilation.entry(getattr(owner, name)), owner=owner)
            assert frontend == {"leaf": count, "parent": count}
            compilation.compile(code)
            live = {native for function in compilation.functions.values()
                    for native in function.taichi_functions.values()}
            assert len(live) == 2
            assert len(compilation._retained_functions) == 2
            assert all(native in live or native._definition_retired() for native in originals)
            originals.extend(live)
            bindings = []
            for capture in code.dynamic:
                field = component.read(owner, capture)
                bindings.append((str(field.dtype), field.shape, field.to_numpy(), field is owner.out))
            snapshots.append((compilation.symbol(code), bindings, expected))
            getattr(owner, name)()
            np.testing.assert_array_equal(owner.out.to_numpy(), expected)
        path = tmp_path / "receiver-reuse.tcm"
        module.archive(str(path))
        del owner, child, other, component, field, code
        gc.collect()
        assert all(reference() is None for reference in owner_refs)
    assert all(native._definition_retired() for native in originals)
    with Runtime("cpu") as runtime:
        loaded = runtime.load_module(str(path))
        for symbol, bindings, expected in snapshots:
            arrays = [runtime.ndarray(dtype, shape) for dtype, shape, _, _ in bindings]
            for array, (_, _, initial, output) in zip(arrays, bindings):
                array.write((np.full_like(initial, -1234) if output else initial).tobytes())
            with loaded.kernel(symbol).bind(*arrays).prepare(specialize=True) as ready:
                ready.submit().wait()
                output = next(array for array, binding in zip(arrays, bindings) if binding[3])
                np.testing.assert_array_equal(np.frombuffer(output.read(), np.int32), expected)
                for array, (_, _, initial, is_output) in zip(arrays, bindings):
                    if not is_output:
                        array.write((initial + 7).tobytes())
                ready.submit().wait()
                np.testing.assert_array_equal(np.frombuffer(output.read(), np.int32), expected + 14)
            for array in arrays:
                array.close()


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_plan_specialization_preserves_control_aliases_and_independent_bindings(tmp_path):
    import numpy as np
    from taichi_forge_runtime.aot import ExecutionPlan, Runtime

    @ti.kernel
    def condition(control: ti.types.ndarray(ti.i32, 1), predicate: ti.types.ndarray(ti.i32, 0)):
        predicate[None] = ti.cast(control[0] > 0, ti.i32)

    @ti.kernel
    def step(source: ti.types.ndarray(ti.i32, 1), out: ti.types.ndarray(ti.i32, 1),
             control: ti.types.ndarray(ti.i32, 1), mode: ti.i32):
        for i in range(out.shape[0]):
            if mode == 3:
                out[i] = source[i] + 3
            else:
                out[i] = source[i] + 7
        control[0] -= 1

    cuda = ti.lang.impl.current_cfg().arch == ti.cuda
    artifact = ti.aot.Module(artifact_format="bc-library" if cuda else "bc")
    artifact.add_kernel(condition)
    artifact.add_kernel(step)
    path = tmp_path / "specialized-plan.tcm"
    artifact.archive(str(path))
    operation = dict(kind="kernel", entry="step", arguments=["source", "out", "control", "mode"],
                     specialization=dict(constant_args=[3]))
    plan = ExecutionPlan([dict(
        kind="while", predicate="predicate", max_iterations=4,
        condition=[dict(kind="kernel", entry="condition", arguments=["control", "predicate"],
                        specialization=dict(constant_args=[]))],
        body=[operation])], resources=[dict(kind="ndarray", name="predicate", dtype="i32", shape=[])])
    with Runtime("cuda" if cuda else "cpu") as runtime:
        module = runtime.load_module(path)
        outputs = [runtime.ndarray("i32", (3,)) for _ in range(2)]
        source = runtime.ndarray("i32", (3,))
        controls = [runtime.ndarray("i32", (1,)) for _ in outputs]
        for array in (*outputs, source):
            array.write(bytes(array.shape[0] * 4))
        for control in controls:
            control.write(np.array([2], np.int32).tobytes())
        arguments = dict(source=outputs[0], out=outputs[0], control=controls[0], mode=Scalar("i32", 3))
        with plan.prepare(module, arguments) as first, plan.prepare(module, dict(
                source=source, out=outputs[1], control=controls[1], mode=Scalar("i32", 3))) as peer:
            arguments["mode"] = Scalar("i32", 7)  # Published binding is a snapshot.
            first.run()
            peer.run()
            runtime.wait()
            np.testing.assert_array_equal(np.frombuffer(outputs[0].read(), np.int32), [6] * 3)
            np.testing.assert_array_equal(np.frombuffer(outputs[1].read(), np.int32), [3] * 3)
            controls[0].write(np.array([1], np.int32).tobytes())
            first.run()
            runtime.wait()
            np.testing.assert_array_equal(np.frombuffer(outputs[0].read(), np.int32), [9] * 3)
            np.testing.assert_array_equal(np.frombuffer(outputs[1].read(), np.int32), [3] * 3)
        # A fresh binding observes both changed scalar facts and array shape.
        resized = runtime.ndarray("i32", (5,))
        resized.write(bytes(20))
        controls[0].write(np.array([2], np.int32).tobytes())
        with plan.prepare(module, dict(source=resized, out=resized, control=controls[0],
                                       mode=Scalar("i32", 7))) as rebound:
            rebound.run()
            runtime.wait()
            np.testing.assert_array_equal(np.frombuffer(resized.read(), np.int32), [14] * 5)
        # This valid ABI binding reaches the backend's scalar specialization
        # validator, proving the per-operation intent was not ignored.
        invalid = ExecutionPlan([dict(operation, specialization=dict(constant_args=[0]))])
        with pytest.raises(RuntimeError, match="integer scalar"):
            invalid.prepare(module, arguments)


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("changing_dtype", [ti.i32, ti.f32, ti.f64])
@pytest.mark.parametrize("ordinary", [False, True])
def test_plan_specialization_keeps_submission_parameters_live(tmp_path, changing_dtype, ordinary):
    from contextlib import nullcontext
    import numpy as np
    from taichi_forge_runtime.aot import ExecutionPlan, Runtime

    @ti.kernel
    def publish(out: ti.types.ndarray(changing_dtype, 1), fixed: ti.i32, changing: changing_dtype):
        for i in range(out.shape[0]):
            out[i] = fixed * 10 + changing

    artifact = ti.aot.Module(artifact_format="bc-library")
    artifact.add_kernel(publish)
    path = tmp_path / "specialized-parameters.tcm"
    artifact.archive(str(path))
    plan = ExecutionPlan([dict(kind="kernel", entry="publish", arguments=["out", "fixed", "changing"],
                               specialization=dict(constant_args=[1, 2]))])
    with Runtime("cuda") as runtime:
        module = runtime.load_module(path)
        dtype = str(changing_dtype)
        numpy_dtype = {"i32": np.int32, "f32": np.float32, "f64": np.float64}[dtype]
        out = runtime.ndarray(dtype, (3,))
        with (nullcontext() if ordinary else runtime.execution_domain(capacity=2)) as domain, plan.prepare(
                module, dict(out=out, fixed=Scalar("i32", 3), changing=Scalar(dtype, 1)),
                execution_domain=domain, mutable_parameters=("changing",)) as call:
            call.initialize()
            values = (9, -4, 13) if dtype == "i32" else (9.25, -4.5, 13.125)
            for value in values:
                with call.submit(values=dict(changing=value)) as ticket:
                    ticket.wait()
                np.testing.assert_array_equal(np.frombuffer(out.read(), numpy_dtype), [30 + value] * 3)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_ordinary_parameter_frames_retain_pending_use_after_ticket_close(tmp_path):
    import ctypes as C
    import threading
    import numpy as np
    from taichi_forge_runtime.aot import ExecutionPlan, Runtime

    @ti.kernel
    def publish(out: ti.types.ndarray(ti.i32, 1), fixed: ti.i32, slot: ti.i32, value: ti.i32):
        out[slot] = fixed * 10 + value

    artifact = ti.aot.Module(artifact_format="bc-library")
    artifact.add_kernel(publish)
    path = tmp_path / "parameter-frames.tcm"
    artifact.archive(str(path))
    plan = ExecutionPlan([dict(kind="kernel", entry="publish", arguments=["out", "fixed", "slot", "value"],
                               specialization=dict(constant_args=[1, 2, 3]))])
    driver = C.WinDLL("nvcuda.dll") if sys.platform == "win32" else C.CDLL("libcuda.so.1")
    callback_type = (C.WINFUNCTYPE if sys.platform == "win32" else C.CFUNCTYPE)(None, C.c_void_p)
    enqueue = driver.cuLaunchHostFunc
    enqueue.argtypes, enqueue.restype = [C.c_void_p, callback_type, C.c_void_p], C.c_int
    push = driver.cuCtxPushCurrent_v2
    push.argtypes, push.restype = [C.c_void_p], C.c_int
    pop = driver.cuCtxPopCurrent_v2
    pop.argtypes, pop.restype = [C.POINTER(C.c_void_p)], C.c_int
    release = threading.Event()
    entered = threading.Event()

    @callback_type
    def gate(_):
        entered.set()
        release.wait(10)  # Bounded failure escape; never invoke CUDA in a callback.

    with Runtime("cuda") as runtime:
        module = runtime.load_module(path)
        out = runtime.ndarray("i32", (3,))
        with runtime.execution_domain(capacity=1) as domain:
            context = domain.info["context"]
        arguments = dict(out=out, fixed=Scalar("i32", 3), slot=Scalar("i32", 0), value=Scalar("i32", 0))
        with plan.prepare(module, arguments, mutable_parameters=("slot", "value")) as first, \
                plan.prepare(module, arguments, mutable_parameters=("slot", "value")) as second:
            first.initialize()
            second.initialize()
            assert push(context) == 0
            try:
                assert enqueue(None, gate, None) == 0
                assert entered.wait(3)
                first.submit(values=dict(slot=0, value=11)).close()
                second.submit(values=dict(slot=1, value=22)).close()
                assert first.submit(values=dict(slot=2, value=99)) is None
                assert second.submit(values=dict(slot=2, value=99)) is None
            finally:
                release.set()
                current = C.c_void_p()
                assert pop(C.byref(current)) == 0
            first._wait_parameter_frame()
            with first.submit(values=dict(slot=2, value=33)) as ticket:
                ticket.wait()
            np.testing.assert_array_equal(np.frombuffer(out.read(), np.int32), [41, 52, 63])


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_runtime_wait_completes_retained_ordinary_prepared_replay(tmp_path):
    import ctypes as C
    from taichi_forge_runtime.aot import Runtime

    @ti.kernel
    def slow(out: ti.types.ndarray(ti.f32, 1), iterations: ti.i32):
        for i in range(out.shape[0]):
            x = out[i]
            for _ in range(iterations):
                x = ti.sin(x) * 0.5 + 0.1
            out[i] = x

    module = ti.aot.Module(artifact_format="bc-library")
    module.add_kernel(slow)
    path = tmp_path / "wait.tcm"
    module.archive(str(path))
    with Runtime("cuda") as runtime:
        loaded = runtime.load_module(str(path))
        out = runtime.ndarray("f32", (1,))
        out.write(bytes(4))
        driver = C.WinDLL('nvcuda.dll') if os.name == 'nt' else C.CDLL('libcuda.so.1')
        def api(name, argtypes):
            call = getattr(driver, name)
            call.argtypes, call.restype = argtypes, C.c_int
            return call
        with runtime.execution_domain(capacity=1, instance_capacity=1) as domain:
            context = domain.info['context']
        push = api('cuCtxPushCurrent_v2', [C.c_void_p])
        pop = api('cuCtxPopCurrent_v2', [C.POINTER(C.c_void_p)])
        create = api('cuEventCreate', [C.POINTER(C.c_void_p), C.c_uint])
        record = api('cuEventRecord', [C.c_void_p, C.c_void_p])
        query = api('cuEventQuery', [C.c_void_p])
        sync = api('cuEventSynchronize', [C.c_void_p])
        destroy = api('cuEventDestroy_v2', [C.c_void_p])
        event = C.c_void_p()
        assert push(context) == 0
        assert create(C.byref(event), 2) == 0
        with loaded.kernel("slow").bind(out, Scalar("i32", 1000000)).prepare() as prepared:
            # Retain the call: wait must complete replay without a destructor
            # or a readback accidentally supplying the missing synchronization.
            # Use a driver event: Runtime.wait also marks its own tickets done,
            # so querying one of those would not independently prove completion.
            try:
                for _ in range(2):
                    prepared.run()
                    assert record(event, None) == 0
                    runtime.wait()
                    assert query(event) == 0
                    assert prepared._handle is not None
            finally:
                assert sync(event) == 0
                assert destroy(event) == 0
                previous = C.c_void_p()
                assert pop(C.byref(previous)) == 0


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("cleanup", [None, False, True])
def test_bitcode_library_shares_real_function_and_validates_dependencies(tmp_path, cleanup):
    import numpy as np
    from zipfile import ZipFile, ZIP_DEFLATED
    from taichi_forge_runtime.aot import Runtime

    @ti.real_func
    def shared(value: ti.f32, bias: ti.types.ndarray(ti.f32, 0)) -> ti.f32:
        x = value
        for _ in range(5):
            x = ti.sin(x) * 0.25 + bias[None]
        return x

    @ti.kernel
    def first(data: ti.types.ndarray(ti.f32, 1), bias: ti.types.ndarray(ti.f32, 0)):
        for i in range(data.shape[0]):
            data[i] = shared(shared(data[i], bias), bias)

    @ti.kernel
    def second(data: ti.types.ndarray(ti.f32, 1), bias: ti.types.ndarray(ti.f32, 0)):
        for i in range(data.shape[0]):
            data[i] = shared(data[i], bias) + 1

    module = ti.aot.Module(artifact_format="bc-library")
    if cleanup is not None:
        module._aot_builder._set_ir_cleanup_for_testing(cleanup)
    module.add_kernel(first)
    module.add_kernel(second)
    artifact = tmp_path / "library.tcm"
    module.archive(str(artifact))
    if cleanup is not False:
        stats = module._aot_builder._streaming_export_statistics()
        assert stats['aot_cleaned_units'] == 1
        assert stats['aot_cleaned_roots'] == 2
        assert stats['aot_unit_finalization_hits'] == 1
        # A second save reuses finalized bodies and recomputes no Pass chain.
        module.archive(str(tmp_path / 'again.tcm'))
        again = module._aot_builder._streaming_export_statistics()
        assert again['aot_cleaned_units'] == 1
        assert again['aot_cleaned_roots'] == 2
        assert again['aot_root_finalization_hits'] == 2
    with ZipFile(artifact) as archive:
        units = [name for name in archive.namelist() if name.startswith("code_units/")]
        assert len(units) == 1  # Actual shared storage, not two equal copies.

    with Runtime("cuda") as runtime:
        loaded = runtime.load_module(str(artifact))
        a = runtime.ndarray("f32", (37,))
        b = runtime.ndarray("f32", (37,))
        bias = runtime.ndarray("f32", ())
        bias_value = np.array(0.5, dtype=np.float32)
        bias.write(bias_value.tobytes())
        for name, reference in (("first", first), ("second", second)):
            source = np.linspace(0, 1, 37, dtype=np.float32)
            expected = source.copy()
            reference(expected, bias_value)
            peer_expected = source + 7
            reference(peer_expected, bias_value)
            a.write(source.tobytes())
            b.write((source + 7).tobytes())
            with loaded.kernel(name).bind(a, bias).prepare(specialize=True) as prepared:
                with loaded.kernel(name).bind(b, bias).prepare(specialize=True) as peer:
                    prepared.run()
                    runtime.wait()
                    np.testing.assert_array_equal(np.frombuffer(b.read(), np.float32), source + 7)
                    peer.run()
                    runtime.wait()
                    np.testing.assert_allclose(np.frombuffer(a.read(), np.float32), expected, rtol=1e-6, atol=1e-6)
                    np.testing.assert_allclose(np.frombuffer(b.read(), np.float32), peer_expected, rtol=1e-6, atol=1e-6)
        a.close()
        b.close()
        bias.close()

    for failure, message in (("missing", "Missing AOT code unit"), ("corrupt", "checksum mismatch")):
        broken = tmp_path / (failure + ".tcm")
        with ZipFile(artifact) as original, ZipFile(broken, "w", ZIP_DEFLATED) as output:
            for name in original.namelist():
                if name == units[0] and failure == "missing":
                    continue
                contents = original.read(name)
                if name == units[0]:
                    contents += b"corrupted"
                output.writestr(name, contents)
        with Runtime("cuda") as runtime:
            with pytest.raises(RuntimeError, match=message):
                runtime.load_module(str(broken)).kernel("first")


def export_execution_context_module(tmp_path, artifact_format, explicit_target, tier):
    @ti.func
    def transform(value: ti.types.vector(2, ti.f32)) -> ti.f32:
        return value[0] + value[1] * 2.0 + 1.0

    @ti.real_func
    def carried_value(value: ti.types.vector(2, ti.f32)) -> ti.f32:
        return transform(value)

    @ti.kernel(opt_level=tier)
    def stage(out: ti.types.ndarray(ti.f32, 1), seed: ti.f32, n: ti.i32):
        carried = carried_value(ti.Vector([seed, seed * 2.0]))
        total = 0.0
        for i in range(n):
            total += carried * (i + 1)
        for j in range(n):
            out[j] = total + j

    fields = [ti.field(ti.f32, shape=64) for _ in range(2)]

    @ti.kernel(opt_level=tier)
    def field_stage(out: ti.types.ndarray(ti.f32, 1), seed: ti.f32, n: ti.i32, target: ti.template()):
        carried = carried_value(ti.Vector([seed, seed * 2.0]))
        total = 0.0
        for i in range(n):
            total += carried * (i + 1)
        for j in target:
            target[j] += total + j
        for j in target:
            out[j] = target[j]

    caps = (
        [ti.DeviceCapability.cuda_compute_capability(ti.lang.impl.get_cuda_compute_capability())]
        if explicit_target
        else None
    )
    module = ti.aot.Module(artifact_format=artifact_format, caps=caps)
    module.add_kernel(stage, name="stage")
    module._aot_builder._add_with_launch_abi("gated_stage", module._exported_kernels["stage"], "cuda_graph_gate_v1")
    module._content.append("kernel:gated_stage")
    output_layout = ti.ndarray(ti.f32, 64)
    for i, field in enumerate(fields):
        module.add_field(f"field_{i}", field)
        module.add_kernel(
            field_stage, name=f"field_stage_{i}", template_args={"out": output_layout, "target": field}
        )
    module.add_execution_plan(
        "public_stage",
        ti.aot.ExecutionPlan(
            [dict(kind="kernel", entry="stage", arguments=["temporary", "seed", "n"]),
             dict(kind="copy", source="temporary", destination="out")],
            resources=[dict(kind="ndarray", name="temporary", dtype="f32", shape=[64]),
                       Scalar("i32", 64).resource("n")],
            initialization=[dict(kind="fill32", destination="temporary", value=0)],
        ),
    )
    module.add_execution_plan(
        "public_field",
        ti.aot.ExecutionPlan(
            [dict(kind="kernel", entry="field_stage_0", arguments=["out", "seed", "n"])],
            resources=[Scalar("i32", 64).resource("n")],
        ),
    )
    directory = tmp_path / "module"
    module.save(directory)
    metadata = json.loads((directory / "aot_metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 22
    assert metadata["kernel_format"] == artifact_format
    if not explicit_target:
        assert {item["key"]: item["value"] for item in metadata["required_caps"]} == {
            "cuda_compute_capability": 80,
            "cuda_ptx_version": 70,
        }
        if artifact_format == "ptx":
            assert b".version 7.0" in (directory / "stage.ptx").read_bytes()
    assert {item["key"]: item["value"] for item in metadata["kernel_launch_abis"]} == {
        "stage": "runtime_context_v2",
        "gated_stage": "cuda_graph_gate_context_v2",
        "field_stage_0": "runtime_context_v2",
        "field_stage_1": "runtime_context_v2",
    }
    assert len(metadata["launch_abi_sha256"]) == 64
    assert {item["key"]: item["value"] for item in metadata["kernel_runtime_state"]} == {
        name: 1 for name in ("stage", "gated_stage", "field_stage_0", "field_stage_1")
    }
    assert len(metadata["runtime_state_sha256"]) == 64
    return directory


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_individual_instance_retirement_before_pending_peer(tmp_path):
    executable = Path(os.environ.get(
        "TI_TEST_CPP_EXECUTABLE",
        str(Path(__file__).resolve().parents[2] / "build" / "taichi_cuda_execution_domain_tests.exe"),
    ))
    if not executable.is_file():
        pytest.skip("Requires the native CUDA execution-domain test executable")
    directory = export_execution_context_module(tmp_path, "bc", False, "fast")
    result = subprocess.run(
        [str(executable), "--gtest_filter=CudaExecutionInstanceRetirement.*:CudaPublicInstanceRetirement.*"],
        env=dict(os.environ, TAICHI_F3_CONTEXT_AOT_FOLDER=str(directory)),
        capture_output=True, text=True, timeout=30,
    )
    (tmp_path / "native-retirement.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode == 0 and "[  SKIPPED ]" in result.stdout:
        pytest.skip("Native instance retirement requires stream-ordered allocation support")
    assert result.returncode == 0 and "[  PASSED  ] 2 tests." in result.stdout, result.stdout + result.stderr


@pytest.mark.parametrize(
    "artifact_format,explicit_target", (("bc", False), ("ptx", False), ("ptx", True), ("cubin", True))
)
@pytest.mark.parametrize("tier", ("fast", "balanced", "full"))
@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_saved_execution_context_isolates_temporary_storage(artifact_format, explicit_target, tier, tmp_path):
    executable = Path(
        os.environ.get(
            "TI_TEST_CPP_EXECUTABLE",
            str(
                Path(__file__).resolve().parents[2]
                / "build"
                / (
                    "taichi_cuda_execution_domain_tests.exe"
                    if os.name == "nt"
                    else "taichi_cuda_execution_domain_tests"
                )
            ),
        )
    )
    if not executable.is_file():
        pytest.skip("Requires the native CUDA execution-domain test executable")
    if artifact_format == "cubin" and not (os.environ.get("TI_CUDA_PTXAS_PATH") or shutil.which("ptxas")):
        pytest.skip("Requires an external ptxas for cubin export")

    directory = export_execution_context_module(tmp_path, artifact_format, explicit_target, tier)
    result = subprocess.run(
        [str(executable), "--gtest_filter=CudaExecutionContextAot.*:CudaPublicExecutionAot.*:CudaExecutionOwner.ExternalStorageLoanPreservesCompleteAotEpoch"],
        env=dict(os.environ, TAICHI_F3_CONTEXT_AOT_FOLDER=str(directory)),
        capture_output=True,
        text=True,
        timeout=90,
    )
    (tmp_path / "native-consumer.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0 and "[  PASSED  ] 11 tests." in result.stdout, result.stdout + result.stderr
    consumer = Path(__file__).with_name("aot_execution_domain_probe.py")
    result = subprocess.run([sys.executable, "-S", str(consumer), str(directory)],
                            env=os.environ.copy(), capture_output=True, text=True, timeout=90)
    (tmp_path / "public-consumer.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0 and "PUBLIC_AOT_DOMAIN_OK" in result.stdout, result.stdout + result.stderr
    external_consumer = Path(__file__).with_name("aot_external_memory_probe.py")
    result = subprocess.run([sys.executable, "-S", str(external_consumer), str(directory)],
                            env=os.environ.copy(), capture_output=True, text=True, timeout=90)
    (tmp_path / "external-consumer.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0 and "PUBLIC_AOT_EXTERNAL_MEMORY_OK" in result.stdout, result.stdout + result.stderr
    from tests.python.aot_execution_domain_probe import check

    with ti.aot.borrow_runtime() as borrower:
        check(borrower, directory)
