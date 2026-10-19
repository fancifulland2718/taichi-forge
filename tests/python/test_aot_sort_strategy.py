"""Export must preserve the producer's storage-dependent sort strategy."""

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.algorithms import _algorithms as algorithms
from tests import test_utils


@pytest.mark.parametrize("field_keys", [False, True])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_mixed_field_ndarray_sort_does_not_change_host_strategy(
    field_keys, monkeypatch
):
    n = 7
    field = ti.field(ti.i32, n)
    ndarray = ti.ndarray(ti.i32, n)
    keys, values = (field, ndarray) if field_keys else (ndarray, field)
    data = np.array([3, 1, 3, -2, 0, 1, -2], dtype=np.int32)
    keys.from_numpy(data)
    values.from_numpy(np.arange(n, dtype=np.int32))
    host = algorithms._host_stable_sort
    calls = []

    def observe(*args, **kwargs):
        calls.append(True)
        return host(*args, **kwargs)

    monkeypatch.setattr(algorithms, "_host_stable_sort", observe)
    sequence = ti.algorithms.primitive_sequence().sort(keys, values)
    builder = ti.graph.GraphBuilder()
    builder.append_native(sequence)
    graph = builder.compile()

    def reject_export():
        module = ti.aot.Module()
        with pytest.raises(
            ti.TaichiRuntimeError, match="producer's native storage strategy"
        ):
            module.add_execution_plan(
                "sort",
                graph,
                source_bindings={"values" if field_keys else "keys": ndarray},
                snapshot_state=True,
            )
        assert not module._content and not module._fields and not module._array_fields

    reject_export()
    assert calls == []
    np.testing.assert_array_equal(keys.to_numpy(), data)
    np.testing.assert_array_equal(values.to_numpy(), np.arange(n, dtype=np.int32))
    sequence.run()
    reject_export()
    assert calls == [True]
    np.testing.assert_array_equal(values.to_numpy(), np.argsort(data, kind="stable"))
    graph.close()
    sequence.clear()
