"""Public aggregate values share ordinary/prepared storage and resource owners."""

import json
import os
from pathlib import Path
import subprocess
import sys

import taichi_forge as ti
import numpy as np
import pytest
from contextlib import ExitStack, closing
from taichi_forge._lib import utils
from tests import test_utils
from tests.cpp.aot.python_scripts.value_argument_aot import export


@test_utils.test(
    arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False, fast_math=False,
    compile_tier="fast", advanced_optimization=False, llvm_opt_level=0,
    external_optimization_level=0, full_simplify_global_iter_cap=1,
)
def test_jit_aot_function_contract_matches_values_effects_and_static_inputs(tmp_path):
    from taichi_forge_runtime.aot import Scalar

    # Share the numerical optimization settings above. CUDA AOT uses an
    # explicit current target instead of its portable SM80 default; the task
    # below declares its own block size, rather than a producer-derived hint.
    caps = ([ti.DeviceCapability.cuda_compute_capability(ti.lang.impl.get_cuda_compute_capability())]
            if ti.cfg.arch == ti.cuda else None)

    @ti.func
    def adjust(value: ti.types.vector(2, ti.i32), positive: ti.template()) -> ti.i32:
        if ti.static(positive):
            value[0] += 7
        else:
            value[0] -= 3
        value[1] *= 2
        return value[0] + value[1]

    @ti.kernel
    def update(data: ti.types.ndarray(ti.i32, 1), ledger: ti.types.ndarray(ti.i32, 1),
               delta: ti.i32, positive: ti.template()):
        ti.loop_config(block_dim=64)
        for i in range(data.shape[0]):
            value = ti.Vector([data[i], delta])
            ledger[i] += adjust(value, positive)
            data[i] = value[0] + value[1]  # adjust must not mutate this value.

    module = ti.aot.Module(caps=caps)
    exemplars = {name: ti.ndarray(ti.i32, shape=(5,)) for name in ("data", "ledger")}
    for positive in (False, True):
        module.add_kernel(update, template_args={**exemplars, "positive": positive},
                          name=f"update_{int(positive)}")
    artifact = tmp_path / "function-contract.tcm"
    module.archive(str(artifact))

    with ti.aot.borrow_runtime() as runtime, ExitStack() as stack:
        loaded = stack.enter_context(runtime.load_module(artifact))
        data = stack.enter_context(closing(runtime.ndarray("i32", (5,))))
        ledger = stack.enter_context(closing(runtime.ndarray("i32", (5,))))
        delta = 4
        for positive in (False, True):
            entry = loaded.kernel(f"update_{int(positive)}")
            with entry.bind(data, ledger, Scalar("i32", delta)).prepare(specialize=True) as prepared:
                # Change ordinary resource contents without re-exporting or
                # preparing again. Both paths must observe the new contents.
                for seed in (1, 9):
                    initial = np.arange(5, dtype=np.int32) + seed
                    initial_ledger = np.full(5, seed * 2, dtype=np.int32)
                    jit_data, jit_ledger = initial.copy(), initial_ledger.copy()
                    update(jit_data, jit_ledger, delta, positive)
                    data.write(initial)
                    ledger.write(initial_ledger)
                    prepared.run()
                    actual_data = np.frombuffer(data.read(), np.int32)
                    actual_ledger = np.frombuffer(ledger.read(), np.int32)
                    expected_ledger = initial_ledger + initial + (7 if positive else -3) + 2 * delta
                    np.testing.assert_array_equal(jit_data, initial + delta)
                    np.testing.assert_array_equal(jit_ledger, expected_ledger)
                    np.testing.assert_array_equal(actual_data, jit_data)
                    np.testing.assert_array_equal(actual_ledger, jit_ledger)

        # The same incompatible resource dtype is rejected through each
        # path's existing argument checks, without requiring equal exceptions.
        with pytest.raises((ValueError, ti.TaichiRuntimeError), match="[Dd]type|[Tt]ype"):
            update(np.zeros(5, dtype=np.float32), np.zeros(5, dtype=np.int32), delta, False)
        wrong = stack.enter_context(closing(runtime.ndarray("f32", (5,))))
        with pytest.raises((ValueError, RuntimeError), match="[Dd]type|[Tt]ype"):
            loaded.kernel("update_0").bind(wrong, ledger, Scalar("i32", delta)).prepare(specialize=True)


@test_utils.test(arch=ti.cpu, cpu_max_num_threads=8, offline_cache=False)
def test_cpu_prepare_specializes_only_fixed_range_chunks(tmp_path):
    from taichi_forge.profiler.cpu_scheduler import (
        query_cpu_scheduler_telemetry, set_cpu_scheduler_telemetry,
    )
    from taichi_forge_runtime.aot import Scalar

    @ti.kernel
    def fill(data: ti.types.ndarray(ti.i32, 1), start: ti.i32, stop: ti.i32,
             active: ti.types.ndarray(ti.i32, 1)):
        for i in range(start, stop):
            data[i] += 1
        for i in range(data.shape[0]):
            data[i] += 2
        for i in range(active[0]):
            data[i] += 4

    module = ti.aot.Module(artifact_format="bc")
    module.add_kernel(fill)
    artifact = tmp_path / "cpu_chunks.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded, ExitStack() as stack:
        entry = loaded.kernel("fill")
        size = 1031
        data = stack.enter_context(closing(runtime.ndarray("i32", (size,))))
        active = stack.enter_context(closing(runtime.ndarray("i32", (1,))))
        active.write(np.zeros(1, np.int32))
        cases = []
        for start, stop, fixed, chunks in (
            (11, 13, (1, 2), 1), (5, 518, (1, 2), 2),
            (3, 3, (1, 2), 0), (10, 9, (1, 2), 0),
            (0, 2, (1,), 8), (0, 1025, (1,), 8),
        ):
            bound = entry.bind(data, Scalar("i32", start), Scalar("i32", stop), active)
            ready = stack.enter_context(closing(bound.prepare(specialize=True, constant_args=fixed)))
            cases.append((ready, start, stop, chunks))
        set_cpu_scheduler_telemetry(True)
        try:
            for ready, start, stop, chunks in reversed(cases):
                active.write(np.array([stop], np.int32))
                data.write(np.zeros(size, np.int32))
                query_cpu_scheduler_telemetry(reset=True)
                ready.run()
                runtime.wait()
                # Shape takes three chunks; the live array load keeps all eight.
                assert query_cpu_scheduler_telemetry()["tasks_requested"] == chunks + 3 + 8
                expected = np.full(size, 2, dtype=np.int32)
                expected[start:stop] += 1
                expected[:stop] += 4
                np.testing.assert_array_equal(np.frombuffer(data.read(), np.int32), expected)
        finally:
            set_cpu_scheduler_telemetry(False, reset=True)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
@pytest.mark.parametrize("fold_dimensions", [False, True])
def test_prepared_bitcode_specialization_keeps_shapes_and_instances_independent(tmp_path, fold_dimensions):
    from taichi_forge_runtime.aot import Scalar

    decorator = ti.func if ti.cfg.arch == ti.vulkan else ti.real_func

    @decorator
    def helper(value: ti.i32) -> ti.i32:
        return 2 * value + 1

    @ti.kernel
    def update(x: ti.types.ndarray(ti.types.vector(2, ti.i32), 2), n: ti.i32, gain: ti.f32):
        for i, j in x:
            if j < n:
                x[i, j][0] = helper(x[i, j][0]) + n
                x[i, j][1] += ti.cast(gain, ti.i32)

    module = ti.aot.Module()
    module.add_kernel(update)
    artifact = tmp_path / "specialized.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime, ExitStack() as stack:
        loaded = stack.enter_context(runtime.load_module(artifact))
        entry = loaded.kernel("update")
        records = []
        # Same shape shares code, not the pointer or contents. Other shapes and
        # explicit scalar constants select distinct, immutable preparations.
        for shape, n, initial, gain in [((3, 7), 5, 1, 3), ((3, 7), 5, 11, 7),
                                       ((4, 9), 8, 21, -2), ((3, 7), 2, 31, 9)]:
            array = stack.enter_context(closing(runtime.ndarray("i32", shape, element_shape=(2,))))
            array.write(np.full((*shape, 2), initial, dtype=np.int32))
            bound = entry.bind(array, Scalar("i32", n), Scalar("f32", float(gain)))
            ready = stack.enter_context(closing(bound.prepare(
                specialize=True, constant_args=(1,) if fold_dimensions else ())))
            records.append((array, ready, shape, n, initial, gain))
        for array, ready, shape, n, initial, gain in reversed(records):
            ready.run()
            expected = np.full((*shape, 2), initial, dtype=np.int32)
            expected[:, :n] += [initial + n + 1, gain]
            np.testing.assert_array_equal(np.frombuffer(array.read(), np.int32).reshape(*shape, 2), expected)
            array.write(np.zeros((*shape, 2), dtype=np.int32))
            ready.run()
            expected.fill(0)
            expected[:, :n] += [n + 1, gain]
            np.testing.assert_array_equal(np.frombuffer(array.read(), np.int32).reshape(*shape, 2), expected)
        with pytest.raises(Exception, match="integer scalar"):
            bound.prepare(specialize=True, constant_args=(2,))
        with pytest.raises(ValueError, match="requires specialize"):
            bound.prepare(constant_args=(1,))


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_prepared_local_tensor_keeps_public_array_writes(tmp_path):
    from taichi_forge_runtime.aot import Scalar

    @ti.kernel
    def update(work: ti.types.ndarray(ti.i32, 2), out: ti.types.ndarray(ti.i32, 1), gain: ti.i32):
        for i in range(out.shape[0]):
            for j in range(work.shape[1]):
                work[i, j] = i + j + gain
            out[i] = work[i, work.shape[1] - 1]

    module = ti.aot.Module()
    module.add_kernel(update)
    archive = tmp_path / "local-tensor.tcm"
    module.archive(str(archive))
    with ti.aot.borrow_runtime() as runtime, ExitStack() as stack:
        entry = stack.enter_context(runtime.load_module(archive)).kernel("update")
        shape = (17, 4)
        scratch = stack.enter_context(closing(runtime.ndarray("i32", shape, usage=("storage", "local_tensor"))))
        public = stack.enter_context(closing(runtime.ndarray("i32", shape)))
        out = stack.enter_context(closing(runtime.ndarray("i32", (shape[0],))))
        # Prime the module cache with private scratch first. The next binding
        # has the same shape and pointer relationships but observable writes.
        for array, gain in ((scratch, 7), (public, 3), (scratch, 2)):
            with closing(entry.bind(array, out, Scalar("i32", gain)).prepare(specialize=True)) as call:
                call.run()
                np.testing.assert_array_equal(np.frombuffer(out.read(), np.int32), np.arange(17) + 3 + gain)
        np.testing.assert_array_equal(np.frombuffer(public.read(), np.int32).reshape(shape),
                                      np.arange(17)[:, None] + np.arange(4) + 3)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
@pytest.mark.parametrize("archive", [False, True])
@pytest.mark.parametrize("artifact_format", ["bc", "bc-library"])
def test_prepared_metadata_reuse_preserves_artifact_snapshot(tmp_path, archive, artifact_format):
    from zipfile import ZipFile, ZIP_DEFLATED
    from taichi_forge_runtime.aot import Scalar

    if artifact_format == "bc-library" and ti.cfg.arch != ti.cuda:
        pytest.skip("bc-library is CUDA-only")

    @ti.kernel
    def update(data: ti.types.ndarray(ti.i32, 1), increment: ti.i32):
        for i in data:
            data[i] += increment

    module = ti.aot.Module(artifact_format=artifact_format)
    module.add_kernel(update)
    artifact = tmp_path / "snapshot.tcm"
    module.archive(str(artifact))
    with ZipFile(artifact) as source:
        entries = {name: source.read(name) for name in source.namelist()}
        source.extractall(tmp_path / "directory")
    path = artifact if archive else tmp_path / "directory"
    with ti.aot.borrow_runtime() as runtime, ExitStack() as stack:
        loaded = stack.enter_context(runtime.load_module(path))
        kernel = loaded.kernel("update")
        data = stack.enter_context(closing(runtime.ndarray("i32", (7,))))
        data.write(np.zeros(7, np.int32))
        entries["metadata.tcb"] += b"changed after loading"
        if archive:
            with ZipFile(artifact, "w", ZIP_DEFLATED) as target:
                for name, contents in entries.items():
                    target.writestr(name, contents)
        else:
            (path / "metadata.tcb").write_bytes(entries["metadata.tcb"])

        def prepare(value):
            return kernel.bind(data, Scalar("i32", value)).prepare(specialize=True, constant_args=(1,))

        if archive:
            # The loaded TCM owns a snapshot. A new specialization and deferred
            # generic registration both use that original contract and code.
            for increment in (2, 3):
                with prepare(increment) as ready:
                    ready.run()
            kernel.run(data, Scalar("i32", 4))
            np.testing.assert_array_equal(np.frombuffer(data.read(), np.int32), np.full(7, 9))
        else:
            with pytest.raises(RuntimeError, match="metadata changed"):
                prepare(2)
            with pytest.raises(RuntimeError, match="metadata changed"):
                kernel.run(data, Scalar("i32", 4))
        # A newly loaded module must not accept the corrupted metadata.
        with pytest.raises(RuntimeError, match="checksum mismatch"):
            runtime.load_module(path)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
@pytest.mark.parametrize("artifact_format", ["bc", "bc-library"])
def test_prepared_real_function_arguments_have_their_own_layout(tmp_path, artifact_format):
    from taichi_forge_runtime.aot import Scalar

    if artifact_format == "bc-library" and ti.cfg.arch != ti.cuda:
        pytest.skip("bc-library is CUDA-only")

    @ti.real_func
    def leaf(value: ti.i32, count: ti.i32) -> ti.i32:
        result = value
        for i in range(count):
            result += i + 1
        return result

    @ti.real_func
    def shared(mode: ti.i32, data: ti.types.ndarray(ti.i32, 1), count: ti.i32):
        for i in range(data.shape[0]):
            # Two calls to one leaf must keep their distinct arguments. The
            # entry and this callee deliberately use different argument orders.
            if mode:
                data[i] = leaf(data[i], count) + leaf(10, count + 1)
            else:
                data[i] = leaf(data[i], count + 2)

    @ti.kernel
    def update(data: ti.types.ndarray(ti.i32, 1), count: ti.i32, mode: ti.i32):
        shared(mode, data, count)

    module = ti.aot.Module(artifact_format=artifact_format)
    module.add_kernel(update)
    artifact = tmp_path / "real-arguments.tcm"
    module.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime, ExitStack() as stack:
        loaded = stack.enter_context(runtime.load_module(artifact))
        from taichi_forge._lib import core

        def loaded_entries():
            return core.query_int64("cuda_artifact_entry_points_loaded")

        before_lookup = loaded_entries() if ti.cfg.arch == ti.cuda else None
        kernel = loaded.kernel("update")
        if before_lookup is not None:
            assert loaded_entries() == before_lookup
        records = []
        for size, count, mode in ((7, 2, 0), (11, 5, 1), (7, 2, 1)):
            initial = np.arange(size, dtype=np.int32)
            expected = initial.copy()
            update(expected, count, mode)
            array = stack.enter_context(closing(runtime.ndarray("i32", (size,))))
            array.write(initial)
            prepared = stack.enter_context(closing(kernel.bind(
                array, Scalar("i32", count), Scalar("i32", mode)).prepare(
                    specialize=True, constant_args=(1, 2))))
            records.append((array, prepared, expected))
        for array, prepared, expected in reversed(records):
            prepared.run()
            np.testing.assert_array_equal(np.frombuffer(array.read(), np.int32), expected)
        # The generic callable is still usable after several specializations.
        # Its mutable input and scalar arguments must not retain their values.
        array, _, _ = records[0]
        generic_entries = None
        for count, mode in ((4, 1), (1, 0)):
            initial = np.arange(7, dtype=np.int32)
            expected = initial.copy()
            update(expected, count, mode)
            array.write(initial)
            kernel.run(array, Scalar("i32", count), Scalar("i32", mode))
            np.testing.assert_array_equal(np.frombuffer(array.read(), np.int32), expected)
            if ti.cfg.arch == ti.cuda:
                after = loaded_entries()
                if generic_entries is not None:
                    assert after == generic_entries
                generic_entries = after


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
@pytest.mark.parametrize("field_storage", [False, True])
def test_prepared_specialization_preserves_storage_relations_and_aliases(tmp_path, field_storage):
    if field_storage and ti.cfg.arch == ti.vulkan:
        pytest.skip("Vulkan does not admit sliced interleaved Field borrowing")

    @ti.kernel
    def update(x: ti.types.ndarray(ti.i32, 1), y: ti.types.ndarray(ti.i32, 1)):
        # Deliberately ordered accesses exercise partial and exact aliases.
        for _ in range(1):
            for i in range(x.shape[0]):
                x[i] += 1
                y[i] += x[i] * 3

    exemplar = ti.ndarray(ti.i32, 16)
    view = ti.experimental.ndarray_view(exemplar, slices=slice(0, 8, 2))
    exported = ti.aot.Module()
    exported.add_kernel(update, template_args={"x": view, "y": view})
    artifact = tmp_path / "storage-relations.tcm"
    exported.archive(str(artifact))
    with ti.aot.borrow_runtime() as runtime, ExitStack() as stack:
        loaded = stack.enter_context(runtime.load_module(artifact))
        records = []

        def make_storage():
            if not field_storage:
                return ti.ndarray(ti.i32, 16), None
            fields = ti.FieldsBuilder()
            values, sentinel = ti.field(ti.i32), ti.field(ti.i32)
            fields.dense(ti.i, 16).place(values, sentinel)
            tree = fields.finalize()
            stack.callback(tree.destroy)
            sentinel.fill(71)
            return values, sentinel

        # Equal layouts on separate allocations, negative offsets, exact and
        # partial aliases, then independent allocations with the same ABI.
        # Shifting both slices preserves their relative layout but changes
        # their byte offsets inside aligned Vulkan descriptor bindings.
        for index, (left, right) in enumerate(((0, 6), (0, 6), (1, 7), (6, 0), (0, 0), (0, 2), (0, None))):
            source, sentinel = make_storage()
            other, other_sentinel = make_storage() if right is None else (source, sentinel)
            initial = np.arange(16, dtype=np.int32) + 100 * index
            source.from_numpy(initial)
            if other is not source:
                other.from_numpy(initial + 50)
            x = stack.enter_context(closing(runtime.import_dense_view(
                ti.experimental.ndarray_view(source, slices=slice(left, left + 4)))))
            begin = 0 if right is None else right
            y = stack.enter_context(closing(runtime.import_dense_view(
                ti.experimental.ndarray_view(other, slices=slice(begin, begin + 4)))))
            ready = stack.enter_context(closing(loaded.kernel("update").bind(x, y).prepare(specialize=True)))
            records.append((source, other, sentinel, other_sentinel, left, begin, ready))

        for _ in range(2):
            for source, other, sentinel, other_sentinel, left, right, ready in reversed(records):
                expected = source.to_numpy()
                expected_other = expected if source is other else other.to_numpy()
                for i in range(4):
                    expected[left + i] += 1
                    expected_other[right + i] += expected[left + i] * 3
                ready.run()
                runtime.wait()
                np.testing.assert_array_equal(source.to_numpy(), expected)
                np.testing.assert_array_equal(other.to_numpy(), expected_other)
                for guard in (sentinel, other_sentinel):
                    if guard is not None:
                        np.testing.assert_array_equal(guard.to_numpy(), np.full(16, 71))


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
@pytest.mark.parametrize("unsigned", [False, True])
@pytest.mark.parametrize("bits", [16, 64])
def test_prepared_specialization_preserves_integer_bits(tmp_path, unsigned, bits):
    from taichi_forge_runtime.aot import Scalar

    name = ("u" if unsigned else "i") + str(bits)
    dtype = getattr(ti, name)

    @ti.kernel
    def update(output: ti.types.ndarray(dtype, 1), fixed: dtype, changing: dtype):
        for i in output:
            output[i] = fixed + changing + ti.cast(i, dtype)

    caps = []
    if ti.cfg.arch == ti.vulkan:
        caps = [f"spirv_has_int{bits}"]
        if bits == 16:
            caps.append("spirv_has_int16_storage")
    module = ti.aot.Module(caps=caps)
    module.add_kernel(update)
    artifact = tmp_path / "wide.tcm"
    module.archive(str(artifact))
    fixed = (1 << (bits - 1)) + 7 if unsigned else -(1 << (bits - 4)) - 7
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(artifact) as loaded, ExitStack() as stack:
        array = stack.enter_context(closing(runtime.ndarray(name, 9)))
        calls = [stack.enter_context(closing(loaded.kernel("update").bind(
            array, Scalar(name, fixed), Scalar(name, changing)).prepare(
                specialize=True, constant_args=(1,)))) for changing in (2, 17)]
        for call, changing in zip(reversed(calls), (17, 2)):
            call.run()
            expected = np.arange(9, dtype=np.dtype(("uint" if unsigned else "int") + str(bits)))
            expected += fixed + changing
            np.testing.assert_array_equal(np.frombuffer(array.read(), dtype=expected.dtype), expected)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_value_arguments_forward_adjoint_and_prepared_retirement(tmp_path):
    arch = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    seconds = export(arch, tmp_path)
    print(json.dumps(dict(arch=arch, export_seconds=seconds,
                          artifact_bytes=(tmp_path / "values.tcm").stat().st_size)))
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=os.pathsep.join([str(root / "packaging/runtime/src"), *sys.path]),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        TI_ENABLE_VERSION_CHECK="0",
    )
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [sys.executable, "-S", str(Path(__file__).with_name("aot_value_argument_probe.py")),
             mode, arch, str(tmp_path / "values.tcm")],
            env=environment, capture_output=True, text=True, timeout=120,
        )
        output = result.stdout + result.stderr
        print(output)
        assert result.returncode == 0, output
        assert "VALUE_ARGUMENTS_PASSED" in output
        assert not any(item in output for item in ("VUID-", "Validation Error", "SYNC-HAZARD")), output
