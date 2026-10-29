"""Runtime-only consumer of normal initialization and numerical replay."""
import importlib.abc
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


with Runtime(sys.argv[2]) as runtime, runtime.load_module(Path(sys.argv[1])) as loaded:
    component = loaded.component("step")
    coordinates = [(i,) for i in range(8)]
    with component.instantiate({}) as left, component.instantiate({}) as right:
        # Consumer inputs are created after freeze, without producer data.
        for instance, base in ((left, 23), (right, 51)):
            instance.resource("initial").write_at(coordinates, struct.pack("<8i", *range(base, base + 8)))
            instance.initialize_state(values=dict(seed=4))

        def advance(instance, bump):
            domain = runtime.execution_domain(capacity=1, instance_capacity=1) if runtime.arch == "cuda" else None
            try:
                with instance.prepare(values=dict(bump=bump), execution_domain=domain) as call:
                    call.initialize()
                    if domain is None:
                        call.run()
                    else:
                        with call.submit() as ticket:
                            ticket.wait()
            finally:
                if domain is not None:
                    domain.close()

        advance(left, 2)
        advance(right, 6)
        advance(left, 3)
        assert struct.unpack("<8i", left.resource("state").read_at(coordinates)) == tuple(range(32, 40))
        left.initialize_state(values=dict(seed=1), reset=True)
        assert struct.unpack("<8i", left.resource("state").read_at(coordinates)) == tuple(range(24, 32))
        left.close()
        advance(right, 2)
        assert struct.unpack("<8i", right.resource("state").read_at(coordinates)) == tuple(range(63, 71))
print("INSTANCE_INITIALIZER_CONSUMER_OK")
