"""Runtime-only Vulkan component state, bounds, initialization and retirement."""

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


with Runtime("vulkan") as runtime, runtime.load_module(sys.argv[1]) as module:
    component = module.component("step")
    before = len(runtime._resources)
    try:
        component.instantiate(dict(capacity=9, active=10))
    except ValueError as error:
        assert "active_capacity" in str(error)
    else:
        raise AssertionError("Invalid active range accepted")
    assert len(runtime._resources) == before
    layout = module.ndarray_layout(component.contract["resources"]["records"]["layout"])
    with (
        component.instantiate(dict(capacity=53, active=47)) as left,
        component.instantiate(dict(capacity=263, active=257)) as right,
    ):
        assert left.resource("alias") is left.resource("records")
        for instance, seed in ((left, 23), (right, 51)):
            instance.initialize(
                zero=["records"],
                data=dict(
                    output=struct.pack(
                        "<" + "i" * instance.resource("output").shape[0],
                        *([-99] * instance.resource("output").shape[0]),
                    )
                ),
            )
            instance.initialize_state(values=dict(seed=seed))
        a = left.prepare(values=dict(bump=2))
        b = right.prepare(values=dict(bump=7))
        kinds = [a.info["execution_kind"], b.info["execution_kind"]]
        a.run()
        b.run()
        a.run()
        # Close while ordinary queue work may still be outstanding. Native
        # resource retirement must complete accepted work before freeing state.
        left.close()
        b.run()
        b.close()
        runtime.wait()
        values = struct.unpack("<263i", right.resource("output").read())
        assert values == tuple([51 + i + 14 + 10 + 2 * (i % 3) for i in range(257)] + [-99] * 6)
        data = right.resource("records").read()
        for i in (0, 127, 128, 256, 257, 262):
            value = struct.unpack_from("<i", data, i * layout.element_stride + layout.member("value").offset)[0]
            assert value == 51 + i + (14 if i < 257 else 0)
        with right.prepare(values=dict(bump=99), dimensions=dict(active=0)) as empty:
            empty.run()
        assert right.resource("output").read() == struct.pack("<263i", *values)
        # A new prepared call must retain state. Reset is explicit and uses the
        # same original initializer on this instance, without touching code.
        right.initialize_state(values=dict(seed=5), reset=True)
        with right.prepare(values=dict(bump=1), dimensions=dict(active=263)) as final:
            final.run()
        runtime.wait()
        result = struct.unpack("<263i", right.resource("output").read())
        assert result == tuple(16 + i + i % 3 for i in range(263))
    report = dict(
        backend=runtime.arch,
        instance_capacities=[53, 263],
        active_counts=[47, 257, 0, 263],
        execution_kinds=kinds,
        scope="array-only Vulkan component; primary queue; no physical submit trace",
    )
Path(sys.argv[2]).write_text(json.dumps(report, indent=2))
print("VULKAN_COMPONENT_CONSUMER_OK")
