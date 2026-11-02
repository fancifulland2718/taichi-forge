"""Native compaction resolves size/scratch from independent component instances."""

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


@ti.data_oriented
class CompactModel:
    def __init__(self, n, form):
        self.n = self.m = self.f = n
        builder = ti.FieldsBuilder() if form != "array" else None
        scope = ti.experimental.ResourceScope(self, builder=builder)
        allocate = scope.ndarray if builder is None else scope.field
        extent = scope.dim("n")
        scope.values = allocate(ti.i32, shape=extent)
        scope.flags = allocate(ti.i32, shape=scope.dim("f"))
        scope.output = allocate(ti.i32, shape=scope.dim("m"))
        scope.count = allocate(ti.i32, shape=1 if builder is None else ())
        scope.ticks = allocate(ti.i32, shape=1)
        if "columns" in form:
            scope.payload = scope.vector_field(3, ti.f32, shape=extent)
            scope.payload_output = scope.vector_field(3, ti.f32, shape=scope.dim("m"))
        if builder:
            self.tree = builder.finalize()
        self.plan = None
        if form in ("array", "field"):
            self.action = ti.algorithms.primitive_sequence().compact(
                self.values, self.flags, self.output, self.count,
                workspace=ti.algorithms.CompactWorkspace(max_items=n),
            )
        else:
            self.plan = (ti.algorithms.prepare_compact_columns(
                ((self.values, self.output), (self.payload, self.payload_output)), self.flags, self.count
            ) if "columns" in form else ti.algorithms.prepare_compact(self.values, self.flags, self.output, self.count))
            self.action = self.plan.record(persistent=form == "persistent_columns")
            if form == "persistent_columns":
                self.plan.close()  # The immutable recording has independent ownership.

    @ti.kernel
    def tick(self):
        self.ticks[0] += 1

    def record(self, builder):
        sequence = builder.create_sequential()
        sequence.dispatch(self.tick)
        sequence.append_native(self.action)
        builder.append(sequence)

    def record_auto(self, builder):
        builder.dispatch(self.tick)
        builder.append_native(self.action, admission="auto")


def _exercise(tmp_path, form, *, auto=False):
    source = CompactModel(7, form)
    root = ti.aot.Component.from_owner(source).compile_graph(source.record_auto if auto else source.record)
    larger = CompactModel(19, form)
    assert root.contract["semantic_key"] == ti.aot.Component.from_owner(larger).compile_graph(larger.record_auto if auto else larger.record).contract["semantic_key"]
    assert "component_native_primitives_v1" in root.contract["required_capabilities"]
    module = ti.aot.Module()
    root.export(module, "compact")
    artifact = tmp_path / "compact.tcm"
    module.archive(str(artifact))
    backend = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    ti.reset()
    with Runtime(backend) as runtime, runtime.load_module(artifact) as loaded:
        deployed = loaded.component("compact")
        domain = runtime.execution_domain() if backend == "cuda" else None
        try:
            with deployed.instantiate(dict(n=11, m=11, f=13)) as wrong:
                wrong.initialize(zero=tuple(root.contract["resources"]))
                before = set(runtime._resources)
                with pytest.raises((RuntimeError, ValueError), match="[Cc]ompact|[Cc]ompaction"):
                    wrong.prepare(execution_domain=domain)
                assert set(runtime._resources) == before
            with deployed.instantiate(dict(n=257, m=263, f=257)) as a, deployed.instantiate(dict(n=519, m=525, f=519)) as b:
                for instance, n in ((a, 257), (b, 519)):
                    instance.initialize(zero=tuple(root.contract["resources"]))
                    values = np.arange(n, dtype=np.int32) * 7
                    # Negative flags also select records, preserving order.
                    flags = np.where(np.arange(n) % 3 == 0, -2, 0).astype(np.int32)
                    instance.resource("values").write(values.tobytes())
                    instance.resource("flags").write(flags.tobytes())
                    instance.resource("output").write(np.full(n + 6, -99, np.int32).tobytes())
                    if "columns" in form:
                        payload = np.column_stack((values, values + 1, values + 2)).astype(np.float32)
                        instance.resource("payload").write(payload.tobytes())
                        instance.resource("payload_output").write(np.full((n + 6, 3), -99, np.float32).tobytes())
                with a.prepare(execution_domain=domain) as pa, b.prepare(execution_domain=domain) as pb:
                    if domain:
                        pa.initialize()
                        pb.initialize()
                        assert pa.info["execution_kind"] == pb.info["execution_kind"] == "cuda_graph"
                        first, second = pa.submit(), pb.submit()
                        second.wait()
                        first.wait()
                        first.close()
                        second.close()
                    else:
                        pa.run()
                        pb.run()
                    for instance, n in ((a, 257), (b, 519)):
                        selected = np.arange(n, dtype=np.int32)[::3] * 7
                        actual = np.frombuffer(instance.resource("output").read(), np.int32)
                        np.testing.assert_array_equal(actual[:len(selected)], selected)
                        np.testing.assert_array_equal(actual[len(selected):], -99)
                        np.testing.assert_array_equal(np.frombuffer(instance.resource("count").read(), np.int32), [len(selected)])
                        if "columns" in form:
                            payload = np.frombuffer(instance.resource("payload_output").read(), np.float32).reshape(-1, 3)
                            np.testing.assert_array_equal(payload[:len(selected)], np.column_stack((selected, selected + 1, selected + 2)))
                            np.testing.assert_array_equal(payload[len(selected):], -99)
                    previous = a.resource("output").read()
                    a.resource("flags").write(np.zeros(257, np.int32).tobytes())
                    if domain:
                        ticket = pa.submit()
                        ticket.wait()
                        ticket.close()
                    else:
                        pa.run()
                    assert a.resource("output").read() == previous
                    np.testing.assert_array_equal(np.frombuffer(a.resource("count").read(), np.int32), [0])
                    np.testing.assert_array_equal(np.frombuffer(b.resource("count").read(), np.int32), [173])
                a.close()
                with b.prepare(execution_domain=domain) as prepared:
                    if domain:
                        prepared.initialize()
                    prepared.run()
                np.testing.assert_array_equal(np.frombuffer(b.resource("ticks").read(), np.int32), [2])
        finally:
            if domain:
                domain.close()


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
@pytest.mark.parametrize("form", ("array", "field"))
def test_component_compact_sequence(tmp_path, form):
    _exercise(tmp_path, form)


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("form", ("prepared", "columns", "persistent_columns"))
def test_component_prepared_compact(tmp_path, form):
    _exercise(tmp_path, form)


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_component_closed_prepared_compact_rejected():
    source = CompactModel(7, "prepared")
    source.plan.close()
    with pytest.raises(ti.TaichiRuntimeError, match="closed"):
        ti.aot.Component.from_owner(source).compile_graph(source.record)


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("persistent", (False, True))
def test_component_prepared_sort_retains_definition_without_source_plan(tmp_path, persistent):
    source = CompactModel(7, "field")
    plan = ti.algorithms.prepare_sort(source.values, source.flags)
    source.action = plan.record(persistent=persistent)
    if persistent:
        plan.close()
    root = ti.aot.Component.from_owner(source).compile_graph(source.record)
    module = ti.aot.Module()
    root.export(module, "sort")
    artifact = tmp_path / "prepared_sort.tcm"
    module.archive(str(artifact))
    ti.reset()
    with Runtime("cuda") as runtime, runtime.load_module(artifact) as loaded:
        with loaded.component("sort").instantiate(dict(n=257, m=257, f=257)) as instance:
            instance.initialize(zero=tuple(root.contract["resources"]))
            keys = (np.arange(257) % 7)[::-1].astype(np.int32)
            values = np.arange(257, dtype=np.int32)
            instance.resource("values").write(keys.tobytes())
            instance.resource("flags").write(values.tobytes())
            with instance.prepare() as prepared:
                prepared.run()
            order = np.argsort(keys, kind="stable")
            np.testing.assert_array_equal(np.frombuffer(instance.resource("values").read(), np.int32), keys[order])
            np.testing.assert_array_equal(np.frombuffer(instance.resource("flags").read(), np.int32), values[order])


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_component_prepared_sort_rejects_bitwise_policy():
    source = CompactModel(7, "field")
    plan = ti.algorithms.prepare_sort(source.values, nan_policy="bitwise")
    source.action = plan.record()
    with pytest.raises(ti.TaichiRuntimeError, match="NaNs-last"):
        ti.aot.Component.from_owner(source).compile_graph(source.record)


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("form", ("prepared", "persistent_columns"))
def test_component_auto_admission_prepared_compact(tmp_path, form):
    _exercise(tmp_path, form, auto=True)


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("wrong_backend", (False, True))
def test_component_auto_admission_rejects_before_conversion(wrong_backend):
    from taichi_forge.graph._native import NativeGraphExecutable, NativeGraphNode, DispatchGraphAction

    @ti.kernel
    def work():
        pass

    key = work._primal.ensure_compiled()
    action = DispatchGraphAction(((work._primal.compiled_kernels[key], ()),), backends=("cpu",))

    class Unqualified(NativeGraphExecutable):
        @property
        def recordable_action(self):
            return action if wrong_backend else None

        def _capture_component(self, capture):
            raise AssertionError("Unqualified provider must not reach symbolic conversion")

    class SourceNode(NativeGraphNode):
        def compile(self):
            return Unqualified()

    source = CompactModel(7, "field")
    def record(builder):
        builder.append_native(SourceNode(), admission="auto")
    with pytest.raises(ti.TaichiRuntimeError, match="Automatic native Graph admission rejected"):
        ti.aot.Component.from_owner(source).compile_graph(record)
