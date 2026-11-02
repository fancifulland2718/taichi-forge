"""Consecutive unique keeps normal kernels and instance-sized private flags."""

import numpy as np
import pytest

import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


@ti.data_oriented
class UniqueModel:
    def __init__(self, n, keyed=True, *, size=None, form="prepared"):
        self.n = n
        self.m = n + 4
        fields = None if form == "array_sequence" else ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=fields)
        extent, output = scope.dim("n"), scope.dim("m")
        allocate = scope.ndarray if fields is None else scope.field
        scope.keys = allocate(ti.u64, shape=extent)
        scope.output = allocate(ti.u64, shape=output)
        scope.count = allocate(ti.i32, shape=1 if fields is None else ())
        scope.ticks = scope.field(ti.i32, shape=())
        if keyed:
            payload_type = ti.types.vector(3, ti.f32) if form == "prepared" else ti.i32
            scope.payload = allocate(payload_type, shape=extent)
            scope.payload_output = allocate(payload_type, shape=output)
        if fields:
            self.tree = fields.finalize()
        if form == "prepared":
            self.plan = (ti.algorithms.prepare_unique_by_key(
                self.keys, self.payload, self.output, self.payload_output, self.count,
                size=size,
            ) if keyed else ti.algorithms.prepare_unique(self.keys, self.output, self.count, size=size))
            self.action = self.plan.record()
        else:
            sequence = ti.algorithms.primitive_sequence()
            self.plan = (sequence.unique_by_key(self.keys, self.payload, self.output, self.payload_output,
                                               self.count, size=size) if keyed else
                         sequence.unique(self.keys, self.output, self.count, size=size))
            self.action = self.plan

    @ti.kernel
    def tick(self):
        self.ticks[None] += 1

    def record(self, builder):
        builder.dispatch(self.tick)
        for _ in range(2):
            builder.append_native(self.action)


def _exercise_unique(tmp_path, keyed, form):
    source = UniqueModel(7, keyed, form=form)
    vector_payload = form == "prepared"
    payload_dtype = np.float32 if vector_payload else np.int32

    def payload(n):
        return (np.arange(n * 3, dtype=payload_dtype).reshape(n, 3) if vector_payload
                else np.arange(n, dtype=payload_dtype))

    def output_payload(n):
        return np.full((n, 3) if vector_payload else n, -99, payload_dtype)
    keys = np.array([5, 5, 2, 2, 5, 5, 5], np.uint64)
    source.keys.from_numpy(keys)
    if keyed:
        source.payload.from_numpy(payload(7))
    source.plan.run()
    assert source.count.to_numpy().reshape(-1)[0] == 3
    np.testing.assert_array_equal(source.output.to_numpy()[:3], [5, 2, 5])
    root = ti.aot.Component.from_owner(source).compile_graph(source.record)
    other = UniqueModel(19, keyed, form=form)
    assert root.contract["semantic_key"] == ti.aot.Component.from_owner(other).compile_graph(other.record).contract["semantic_key"]
    assert "component_native_private_arrays_v1" in root.contract["required_capabilities"]
    assert len(root.contract["private_shapes"]) == 2
    assert all(shape[0]["array"] in root.contract["bindings"] for shape in root.contract["private_shapes"].values())
    module = ti.aot.Module()
    root.export(module, "unique")
    artifact = tmp_path / "unique.tcm"
    module.archive(str(artifact))
    backend = "cpu" if ti.cfg.arch == ti.cpu else "cuda"
    ti.reset()
    with Runtime(backend) as runtime, runtime.load_module(artifact) as loaded:
        deployed = loaded.component("unique")
        from contextlib import nullcontext

        with (runtime.execution_domain() if backend == "cuda" else nullcontext()) as domain:
            with deployed.instantiate(dict(n=257, m=261)) as a, deployed.instantiate(dict(n=519, m=523)) as b:
                for instance, n in ((a, 257), (b, 519)):
                    instance.initialize(zero=tuple(root.contract["resources"]))
                    keys = ((np.arange(n) // 3) % 5).astype(np.uint64)
                    instance.resource("keys").write(keys.tobytes())
                    instance.resource("output").write(np.full(n + 4, 999, np.uint64).tobytes())
                    if keyed:
                        instance.resource("payload").write(payload(n).tobytes())
                        instance.resource("payload_output").write(output_payload(n + 4).tobytes())
                with a.prepare(execution_domain=domain) as pa, b.prepare(execution_domain=domain) as pb:
                    if domain:
                        pa.initialize()
                        pb.initialize()
                        assert pa.info["execution_kind"] == pb.info["execution_kind"] == "cuda_graph"
                        one, two = pa.submit(), pb.submit()
                        two.wait()
                        one.wait()
                        one.close()
                        two.close()
                    else:
                        pa.run()
                        pb.run()
                    for instance, n in ((a, 257), (b, 519)):
                        keys = ((np.arange(n) // 3) % 5).astype(np.uint64)
                        indices = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
                        actual = np.frombuffer(instance.resource("output").read(), np.uint64)
                        np.testing.assert_array_equal(actual[:len(indices)], keys[indices])
                        np.testing.assert_array_equal(actual[len(indices):], 999)
                        np.testing.assert_array_equal(np.frombuffer(instance.resource("count").read(), np.int32), [len(indices)])
                        if keyed:
                            actual = np.frombuffer(instance.resource("payload_output").read(), payload_dtype)
                            if vector_payload:
                                actual = actual.reshape(-1, 3)
                            np.testing.assert_array_equal(actual[:len(indices)], payload(n)[indices])
                            np.testing.assert_array_equal(actual[len(indices):], -99)
                    a.resource("keys").write(np.full(257, 42, np.uint64).tobytes())
                    if domain:
                        ticket = pa.submit()
                        ticket.wait()
                        ticket.close()
                    else:
                        pa.run()
                    np.testing.assert_array_equal(np.frombuffer(a.resource("count").read(), np.int32), [1])
                    np.testing.assert_array_equal(np.frombuffer(b.resource("count").read(), np.int32), [173])
                a.close()
                with b.prepare(execution_domain=domain) as prepared:
                    if domain:
                        prepared.initialize()
                    prepared.run()
                np.testing.assert_array_equal(np.frombuffer(b.resource("ticks").read(), np.int32), [2])


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("keyed", (False, True))
def test_component_unique_new_extents_and_private_instances(tmp_path, keyed):
    _exercise_unique(tmp_path, keyed, "prepared")


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
@pytest.mark.parametrize("keyed", (False, True))
@pytest.mark.parametrize("form", ("field_sequence", "array_sequence"))
def test_component_unique_sequence_fallback(tmp_path, keyed, form):
    _exercise_unique(tmp_path, keyed, form)


@test_utils.test(arch=ti.cuda, offline_cache=False)
@pytest.mark.parametrize("size", (0, 7))
def test_component_unique_keeps_explicit_prefix_and_rejects_small_extent(tmp_path, size):
    source = UniqueModel(7, keyed=False, size=size)
    root = ti.aot.Component.from_owner(source).compile_graph(source.record)
    larger = UniqueModel(19, keyed=False, size=size)
    assert root.contract["semantic_key"] == ti.aot.Component.from_owner(larger).compile_graph(larger.record).contract["semantic_key"]
    module = ti.aot.Module()
    root.export(module, "prefix")
    artifact = tmp_path / "prefix.tcm"
    module.archive(str(artifact))
    source.plan.close()
    with pytest.raises(ti.TaichiRuntimeError, match="closed"):
        ti.aot.Component.from_owner(source).compile_graph(source.record)
    ti.reset()
    with Runtime("cuda") as runtime, runtime.load_module(artifact) as loaded:
        component = loaded.component("prefix")
        if size:
            before = set(runtime._resources)
            with pytest.raises(ValueError, match="native_prefix"):
                component.instantiate(dict(n=5, m=9))
            assert set(runtime._resources) == before
        with component.instantiate(dict(n=257, m=261)) as instance:
            instance.initialize(zero=tuple(root.contract["resources"]))
            keys = np.arange(257, dtype=np.uint64) // 3
            instance.resource("keys").write(keys.tobytes())
            instance.resource("output").write(np.full(261, 999, np.uint64).tobytes())
            with instance.prepare() as prepared:
                prepared.run()
            count = 3 if size else 0
            np.testing.assert_array_equal(np.frombuffer(instance.resource("count").read(), np.int32), [count])
            output = np.frombuffer(instance.resource("output").read(), np.uint64)
            np.testing.assert_array_equal(output[:count], np.arange(count))
            np.testing.assert_array_equal(output[count:], 999)


@ti.data_oriented
class CandidateSequenceModel:
    def __init__(self, n):
        self.n = n
        fields = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=fields)
        extent = scope.dim("n")
        for name in ("raw", "compact", "unique"):
            setattr(scope, name, scope.field(ti.u64, shape=extent))
        for name in ("flags", "payload", "compact_payload", "unique_payload"):
            setattr(scope, name, scope.field(ti.i32, shape=extent))
        for name in ("compact_count", "payload_count", "unique_count"):
            setattr(scope, name, scope.field(ti.i32, shape=()))
        self.tree = fields.finalize()
        self.sequence = (ti.algorithms.primitive_sequence()
            .compact(self.raw, self.flags, self.compact, self.compact_count)
            .compact(self.payload, self.flags, self.compact_payload, self.payload_count)
            .sort_by_key(self.compact, self.compact_payload)
            .unique_by_key(self.compact, self.compact_payload, self.unique, self.unique_payload, self.unique_count))

    @ti.kernel
    def reset_outputs(self):
        for i in range(self.n):
            self.payload[i] = i
            self.compact[i] = ti.u64(0xFFFFFFFFFFFFFFFF)
            self.unique[i] = ti.u64(0xFFFFFFFFFFFFFFFF)
            self.compact_payload[i] = -1
            self.unique_payload[i] = -1

    def record(self, builder):
        builder.dispatch(self.reset_outputs)
        builder.append_native(self.sequence)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_component_compact_sort_unique_sequence_preserves_order_and_empty_packet(tmp_path):
    source = CandidateSequenceModel(17)
    root = ti.aot.Component.from_owner(source).compile_graph(source.record)
    module = ti.aot.Module()
    root.export(module, "candidate")
    artifact = tmp_path / "candidate.tcm"
    module.archive(str(artifact))
    backend = "cpu" if ti.cfg.arch == ti.cpu else "cuda"
    ti.reset()
    with Runtime(backend) as runtime, runtime.load_module(artifact) as loaded:
        with loaded.component("candidate").instantiate(dict(n=513)) as instance:
            instance.initialize(zero=tuple(root.contract["resources"]))
            keys = ((513 - np.arange(513)) % 11).astype(np.uint64)
            flags = (np.arange(513) % 3 != 0).astype(np.int32)
            instance.resource("raw").write(keys.tobytes())
            instance.resource("flags").write(flags.tobytes())
            with instance.prepare() as prepared:
                prepared.run()
                selected = np.flatnonzero(flags)
                selected = selected[np.argsort(keys[selected], kind="stable")]
                heads = np.r_[0, np.flatnonzero(keys[selected[1:]] != keys[selected[:-1]]) + 1]
                expected = selected[heads]
                actual = np.frombuffer(instance.resource("unique_payload").read(), np.int32)
                np.testing.assert_array_equal(actual[:len(expected)], expected)
                np.testing.assert_array_equal(actual[len(expected):], -1)
                np.testing.assert_array_equal(np.frombuffer(instance.resource("unique_count").read(), np.int32), [len(expected) + 1])
                instance.resource("flags").write(np.zeros(513, np.int32).tobytes())
                prepared.run()
                np.testing.assert_array_equal(np.frombuffer(instance.resource("compact_count").read(), np.int32), [0])
                np.testing.assert_array_equal(np.frombuffer(instance.resource("unique_count").read(), np.int32), [1])
                np.testing.assert_array_equal(np.frombuffer(instance.resource("unique").read(), np.uint64), np.uint64(0xFFFFFFFFFFFFFFFF))
