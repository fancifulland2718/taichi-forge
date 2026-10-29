"""Frozen Dense component consumption without numerical DSL source."""

import importlib.abc
import json
import os
from pathlib import Path
import struct
import sys


class NoNumericalSource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("taichi", "taichi_forge", "numpy", "tests"):
            raise AssertionError("Numerical source imported: " + fullname)


sys.path.insert(0, os.environ["FORGE_AOT_TEST_PACKAGE"])
sys.meta_path.insert(0, NoNumericalSource())
from taichi_forge_runtime.aot import Runtime


def integers(instance, name):
    data = instance.resource(name).read()
    return struct.unpack("<" + "i" * (len(data) // 4), data)


with Runtime("vulkan") as runtime, runtime.load_module(sys.argv[1]) as module:
    component = module.component("dense")
    with (
        component.instantiate(dict(capacity=53, active=47)) as left,
        component.instantiate(dict(capacity=263, active=257)) as right,
    ):
        for instance, seed in ((left, 23), (right, 51)):
            instance.initialize(zero=["x", "y", "v", "tally"])
            instance.initialize_state(values=dict(seed=seed))
            assert integers(instance, "tally") == (0,)
        a = left.prepare(values=dict(bump=2))
        b = right.prepare(values=dict(bump=7))
        a.run()
        b.run()
        a.run()
        left.close()
        b.run()
        b.close()
        assert integers(right, "x") == tuple(51 + i + (14 if i < 257 else 0) for i in range(263))
        assert integers(right, "y") == tuple([2 * (65 + i) for i in range(257)] + [-99] * 6)
        assert integers(right, "v") == (11, -2) * 263
        assert integers(right, "tally") == (514,)
        with right.prepare(values=dict(bump=0), dimensions=dict(active=0)) as empty:
            empty.run()
        assert integers(right, "tally") == (514,)
        assert integers(right, "v") == (11, -1) * 263
        right.initialize_state(values=dict(seed=5), reset=True)
        with right.prepare(values=dict(bump=1), dimensions=dict(active=263)) as full:
            full.run()
        assert integers(right, "y") == tuple(2 * (6 + i) for i in range(263))
        assert integers(right, "tally") == (263,)
    report = dict(
        backend=runtime.arch,
        capacities=[53, 263],
        active_counts=[47, 257, 0, 263],
        scope="One loaded module; original Dense initializer/step; primary-queue cold retirement",
    )
Path(sys.argv[2]).write_text(json.dumps(report, indent=2))
print("VULKAN_DENSE_CONSUMER_OK")
