"""Bounded integration tests for serial integer prelude sinking."""

import taichi_forge as ti
import numpy as np
from tests import test_utils
from tests.python.serial_integer_prelude_probe import run_classify, run_guards


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan],
                 offline_cache=False, advanced_optimization=False, opt_level=1)
def test_interval_classification_jit_graph_aot(tmp_path):
    result = run_classify(tmp_path)
    assert len(result["checks"]) == 15
    assert result["aot_producer_isolation"]


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False,
                 advanced_optimization=False, opt_level=1)
def test_snapshot_rejections_and_ad_fallback(tmp_path):
    result = run_guards(tmp_path)
    assert set(result) == {"dynamic_range", "source_write", "external_alias", "forward_ad", "reverse_ad"}
    assert all(case["passed"] for case in result.values())


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False,
                 advanced_optimization=False, opt_level=1)
def test_byvalue_integer_prefix_uses_one_range():
    out = ti.field(ti.i32, shape=37)

    @ti.kernel
    def fill(n: ti.i32):
        value = ti.min(n, 37)
        for i in range(37):
            out[i] = value + i

    assert [task.task_type for task in fill.task_manifest(11)] == ["range_for"]
    for count in (-3, 0, 11, 37, 57):
        fill(count)
        np.testing.assert_array_equal(out.to_numpy(), np.arange(37) + min(count, 37))
