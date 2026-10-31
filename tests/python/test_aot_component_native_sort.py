"""Native sort shares instance arrays, ordering and existing Runtime scratch."""

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge.aot._component import Component
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


@ti.data_oriented
class SortModel:
    def __init__(self, n):
        self.n = n
        self.m = n
        scope = ti.experimental.ResourceScope(self)
        extent = scope.dim("n")
        scope.keys = scope.ndarray(ti.f32, shape=extent)
        scope.values = scope.ndarray(ti.i32, shape=scope.dim("m"))
        scope.output = scope.ndarray(ti.i32, shape=extent)
        self.workspace = ti.algorithms.SortWorkspace(max_items=n)

    @ti.kernel
    def seed(self):
        for i in range(self.n):
            self.keys[i] = ti.cast((self.n - i) % 7, ti.f32)
            self.values[i] = i

    @ti.kernel
    def observe(self):
        for i in range(self.n):
            self.output[i] = self.values[i]

    def record(self, builder):
        builder.dispatch(self.seed)
        builder.append_native(
            ti.algorithms.primitive_sequence().sort(
                self.keys, self.values, stable=True, workspace=self.workspace
            ),
            prewarm=True,
        )
        builder.dispatch(self.observe)

    def nested(self, builder):
        sequence = builder.create_sequential()
        self.record(sequence)
        builder.append(sequence)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
@pytest.mark.parametrize("nested", (False, True))
def test_native_sort_rebinds_shape_and_isolates_instance_scratch(tmp_path, nested):
    source = SortModel(7)
    component = Component.from_owner(source)
    root = component.compile_graph(source.nested if nested else source.record)
    larger = SortModel(19)
    other = Component.from_owner(larger).compile_graph(larger.nested if nested else larger.record)
    assert root.contract["semantic_key"] == other.contract["semantic_key"]
    assert "component_native_sort_v1" in root.contract["required_capabilities"]
    assert root.contract["resources"]["keys"]["access"] == "read_write"
    module = ti.aot.Module()
    root.export(module, "sort")
    artifact = tmp_path / "sort.tcm"
    module.archive(str(artifact))
    backend = "cpu" if ti.cfg.arch == ti.cpu else "cuda"
    ti.reset()
    with Runtime(backend) as runtime:
        loaded = runtime.load_module(artifact)
        deployed = loaded.component("sort")
        domain = runtime.execution_domain() if backend == "cuda" else None
        with deployed.instantiate(dict(n=11, m=13)) as mismatch:
            mismatch.initialize(zero=("keys", "values", "output"))
            before = set(runtime._resources)
            with pytest.raises((RuntimeError, ValueError), match="Sort payload.*matching"):
                mismatch.prepare(execution_domain=domain)
            assert set(runtime._resources) == before
        with (
            deployed.instantiate(dict(n=257, m=257)) as a,
            deployed.instantiate(dict(n=519, m=519)) as b,
        ):
            for instance in (a, b):
                instance.initialize(zero=("keys", "values", "output"))
            with a.prepare(execution_domain=domain) as pa, b.prepare(execution_domain=domain) as pb:
                if domain:
                    pa.initialize()
                    pb.initialize()
                    assert pa.info["execution_kind"] == "cuda_graph"
                    first, second = pa.submit(), pb.submit()
                    assert first is not None and second is not None
                    second.wait()
                    first.close()
                    second.close()
                    repeated = pa.submit()
                    assert repeated is not None
                    repeated.wait()
                    repeated.close()
                else:
                    pa.run()
                    pb.run()
                    pa.run()
                for instance, n in ((a, 257), (b, 519)):
                    keys = ((n - np.arange(n)) % 7).astype(np.float32)
                    order = np.argsort(keys, kind="stable").astype(np.int32)
                    np.testing.assert_array_equal(
                        np.frombuffer(instance.resource("output").read(), np.int32), order
                    )
                    np.testing.assert_array_equal(
                        np.frombuffer(instance.resource("keys").read(), np.float32), keys[order]
                    )
            a.close()
            with b.prepare(execution_domain=domain) as pb:
                if domain:
                    pb.initialize()
                    ticket = pb.submit()
                    assert ticket is not None
                    ticket.wait()
                    ticket.close()
                else:
                    pb.run()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_native_sort_rejects_undeclared_storage():
    source = SortModel(7)
    source.keys = ti.ndarray(ti.f32, 7)
    with pytest.raises(ti.TaichiCompilationError):
        Component.from_owner(source).compile_graph(source.record)
