"""Frozen consumers prepare nested model inputs from public declaration paths."""
import json
from zipfile import ZipFile

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.aot._component import Component
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


@ti.data_oriented
class _Child:
    def __init__(self, n):
        self.n, self.gain = n, 3
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        scope.value("gain", ti.i32)
        scope.seed = scope.field(ti.i32, shape=scope.dim("n"))
        scope.state = scope.field(ti.i32, shape=scope.dim("n"), initialization="zero")
        self.tree = builder.finalize()
        scope.extra = scope.ndarray(ti.i32, shape=scope.dim("n"))

    @ti.kernel
    def step(self):
        for i in range(self.n):
            self.state[i] += self.seed[i] * self.gain + self.extra[i]


@ti.data_oriented
class _Model:
    def __init__(self, n):
        self.offset = 4
        scope = ti.experimental.ResourceScope(self)
        scope.value("offset", ti.i32)
        scope.child = _Child(n)
        scope.shared = self.child.seed
        # An unrelated declaration must not become an input of this root.
        self.unused = 9
        scope.value("unused", ti.i32)

    @ti.kernel
    def adjust(self):
        for i in self.shared:
            self.shared[i] += self.offset

    def record(self, builder):
        builder.dispatch(self.child.step)
        builder.dispatch(self.adjust)


def _read_path(model, source):
    assert source["kind"] == "attribute"
    value = model
    for part in source["path"]:
        value = value[part]
    return value


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_frozen_nested_inputs_use_public_paths_not_encoded_ids(tmp_path):
    owner = _Model(3)
    root = Component.from_owner(owner).compile_graph(owner.record)
    module = ti.aot.Module()
    root.export(module, "step")
    artifact = tmp_path / "inputs.tcm"
    module.archive(str(artifact))
    backend = "cuda" if ti.cfg.arch == ti.cuda else "cpu"
    ti.reset()
    with Runtime(backend) as runtime, runtime.load_module(artifact) as loaded:
        component = loaded.component("step")
        sources = component.input_sources()
        assert {tuple(source["path"]) for source in sources["dimensions"].values()} == {("child", "n")}
        assert {tuple(source["path"]) for source in sources["values"].values()} == {("child", "gain"), ("offset",)}
        seed_id = next(name for name, aliases in sources["resources"].items()
                       if any(source["path"] == ["shared"] for source in aliases.values()))
        assert {tuple(source["path"]) for source in sources["resources"][seed_id].values()} == {
            ("shared",), ("child", "seed")}
        assert any(len(schema) == 5 for schema in component.contract["dimensions"].values())
        for n, gain, offset in ((11, 2, 5), (17, 7, 8)):
            seed = np.arange(n, dtype=np.int32)
            extra = np.full(n, 9, dtype=np.int32)
            model = dict(child=dict(n=n, gain=gain, seed=seed, extra=extra), shared=seed, offset=offset)
            dimensions = {name: _read_path(model, source) for name, source in sources["dimensions"].items()}
            values = {name: _read_path(model, source) for name, source in sources["values"].items()}
            payloads = {name: _read_path(model, next(iter(aliases.values()))).tobytes()
                        for name, aliases in sources["resources"].items()
                        if component.contract["resources"][name]["initialization"] == "consumer_explicit"}
            state_id = next(name for name, aliases in sources["resources"].items()
                            if any(source["path"] == ["child", "state"] for source in aliases.values()))
            with component.instantiate(dimensions) as instance:
                instance.initialize(payloads)
                with instance.prepare(values=values) as prepared:
                    prepared.run()
                    prepared.run()
                result = np.frombuffer(instance.resource(state_id).read(), np.int32)
                np.testing.assert_array_equal(result, (2 * seed + offset) * gain + 18)
        sources["values"].clear()
        assert component.input_sources()["values"]

    if backend == "cpu":
        with ZipFile(artifact) as archive:
            payloads = {name: archive.read(name) for name in archive.namelist()}
        for case in ("dimensions", "aliases", "path", "legacy"):
            contract = json.loads(payloads["components/step.json"])
            sources = contract["compilation"]["input_sources"]
            if case == "dimensions":
                sources["dimensions"].clear()
            elif case == "aliases":
                sources["resources"][seed_id].popitem()
            elif case == "path":
                next(iter(sources["values"].values()))["path"] = ["not.an.attribute"]
            else:
                contract["compilation"].pop("input_sources")
            path = tmp_path / f"{case}.tcm"
            with ZipFile(path, "w") as archive:
                for name, payload in payloads.items():
                    archive.writestr(name, json.dumps(contract) if name == "components/step.json" else payload)
            with Runtime(backend) as runtime, runtime.load_module(path) as loaded:
                if case == "legacy":
                    with pytest.raises(ValueError, match="rebuild"):
                        loaded.component("step").input_sources()
                else:
                    with pytest.raises(ValueError, match="input source"):
                        loaded.component("step")
