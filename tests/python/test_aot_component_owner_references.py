"""Non-owning numerical links preserve one declared ownership tree and ABI."""
import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


@ti.func
def _forward_record(owner, i):
    alias = owner
    alias.increment(i)


@ti.data_oriented
class _Stage:
    def __init__(self, motion):
        ti.experimental.ResourceScope(self).owner_reference("motion", motion)

    @ti.kernel
    def step(self):
        for i in range(self.motion.n):
            _forward_record(self.motion, i)


@ti.data_oriented
class _Motion:
    def __init__(self, n):
        self.n = n
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.data = scope.field(ti.i32, shape=scope.dim("n"), initialization="zero")
        self.tree = builder.finalize()
        scope.stage = _Stage(self)
        scope.owner_reference("stage_link", self.stage)

    @ti.func
    def increment(self, i):
        self.data[i] += 1

    @ti.kernel
    def direct(self):
        for i in self.data:
            self.data[i] += 2

    @ti.kernel
    def via_link(self):
        for i in range(self.stage_link.motion.n):
            self.stage_link.motion.increment(i)

    def record(self, builder):
        builder.dispatch(self.stage.step)
        builder.dispatch(self.via_link)


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_owner_references_reuse_code_and_reject_retargeted_live_owners(tmp_path):
    first, second = _Motion(3), _Motion(7)
    # Ordinary JIT still traverses the original objects.
    first.stage.step()
    np.testing.assert_array_equal(first.data.to_numpy(), np.ones(3))
    env = ti.aot.Component.from_owner(first)
    other = ti.aot.Component.from_owner(second)
    code = env.compile(first.stage.step, owner=first)
    assert other.compile(second.stage.step, owner=second) is code
    code.run(other, second)
    np.testing.assert_array_equal(second.data.to_numpy(), np.ones(7))
    first.stage.motion = second
    with pytest.raises(ValueError, match="bindings do not match"):
        code.run(env, first)
    with pytest.raises(ti.TaichiCompilationError, match="owner reference.*changed"):
        env.compile(first.stage.step, owner=first)
    with pytest.raises(ti.TaichiCompilationError, match="ownership path.*found 0"):
        ti.aot.Component.from_owner(first)
    first.stage.motion = first
    root = env.compile_graph(first.record)
    assert len(root.contract["groups"]) == 1
    assert set(root.contract["resources"]) == {"data"}
    module = ti.aot.Module()
    root.export(module, "step")
    artifact = tmp_path / "references.tcm"
    module.archive(str(artifact))
    arch = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(arch) as runtime:
        component = runtime.load_module(artifact).component("step")
        assert component.resource_name(["data"]) == "data"
        with component.instantiate({"n": 37}) as instance:
            with instance.prepare() as call:
                call.run()
                runtime.wait()
            np.testing.assert_array_equal(np.frombuffer(instance.resource("data").read(), np.int32), np.full(37, 2))


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_unused_reference_does_not_pollute_kernel_identity_or_allow_ambiguous_ownership():
    first, second = _Motion(3), _Motion(7)
    first_scope, second_scope = ti.experimental.ResourceScope(first), ti.experimental.ResourceScope(second)
    first_scope.owner_reference("unused", first)
    second_scope.owner_reference("unused", second.stage)
    env, other = ti.aot.Component.from_owner(first), ti.aot.Component.from_owner(second)
    code = env.compile(first.direct)
    assert other.compile(second.direct) is code
    assert "owner_references" not in code.contract
    second_scope.duplicate = second.stage
    with pytest.raises(ti.TaichiCompilationError, match="ownership path.*found 2"):
        ti.aot.Component.from_owner(second)
    with pytest.raises(ValueError, match="another declaration role"):
        first_scope.owner_reference("data", first)
    with pytest.raises(TypeError, match="non-null"):
        first_scope.owner_reference("absent", None)
