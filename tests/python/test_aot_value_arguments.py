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


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_prepared_real_function_arguments_have_their_own_layout(tmp_path):
    from taichi_forge_runtime.aot import Scalar

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

    module = ti.aot.Module(artifact_format="bc-library" if ti.cfg.arch == ti.cuda else "bc")
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
        for index, (left, right) in enumerate(((0, 6), (0, 6), (6, 0), (0, 0), (0, 2), (0, None))):
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
