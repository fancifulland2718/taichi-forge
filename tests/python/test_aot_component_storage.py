"""Shared artifact records preserve full contracts and instance admission."""
from copy import deepcopy
import json
from pathlib import Path
import random
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
import pytest
import taichi_forge as ti
import taichi_forge_runtime
from taichi_forge.aot import Component, RootOption, StaticValue
from taichi_forge.aot._component import Record
from taichi_forge_runtime.aot import Runtime, inspect_component, ExecutionPlan
from taichi_forge_runtime.aot._component_info import _document_reader
from taichi_forge_runtime.aot._component_storage import (
    decode_contract_document, encode_contract_documents, decode_plan_document, encode_plan_documents,
)
from tests import test_utils
from tests.python.test_aot_component_configuration import _NestedConfiguration
from tests.python.test_aot_component_dense import _environment


def _documents():
    # A long compiler record separates repeated occurrences beyond deflate's
    # window. Keep the same strings at every occurrence, with nested metadata
    # that must be restored byte-exactly and independently mutable.
    rng = random.Random(123)
    record = {"tasks": [{"capture": rng.randbytes(25000).hex()}], "effects": {"write": ["x"]}}
    contract = {"entries": [deepcopy(record), deepcopy(record)], "name": "first"}
    original = {"components/first.json": contract,
                "components/second.json": {**deepcopy(contract), "name": "second"}}
    return original, encode_contract_documents(original)


def test_shared_records_are_exact_and_do_not_alias():
    original, encoded = _documents()
    assert len(encoded) == 3  # Two public roots reference one complete record.
    for name, expected in original.items():
        restored = decode_contract_document(json.loads(encoded[name]), encoded.__getitem__)
        assert restored == expected
        restored["entries"][0]["effects"]["write"].append("changed")
        assert restored["entries"][1]["effects"] == {"write": ["x"]}
        assert decode_contract_document(json.loads(encoded[name]), encoded.__getitem__) == expected
    # Header/file overhead exceeds sharing savings for tiny contracts.
    tiny = {"components/a.json": {"entries": [{"x": 1}] * 2}}
    assert json.loads(encode_contract_documents(tiny)["components/a.json"]) == tiny["components/a.json"]


@pytest.mark.parametrize("mutation, message", [
    (lambda doc: doc.update(storage_format="future"), "Unsupported.*storage"),
    (lambda doc: doc["references"][0].__setitem__(1, "../../outside"), "record reference"),
    (lambda doc: doc["references"][0].__setitem__(0, ["entries", -1]), "record path"),
    (lambda doc: doc["references"][0].__setitem__(0, ["entries", True]), "record reference"),
    (lambda doc: doc["references"].append(doc["references"][0]), "exactly once"),
    (lambda doc: doc["contract"].update(name="tampered"), "contract digest"),
    (lambda doc: doc["references"].pop(), "contract digest"),
])
def test_invalid_shared_contracts_rejected(mutation, message):
    _, encoded = _documents()
    doc = json.loads(encoded["components/first.json"])
    mutation(doc)
    with pytest.raises(ValueError, match=message):
        decode_contract_document(doc, encoded.__getitem__)


def test_record_integrity_and_archive_membership(tmp_path):
    _, encoded = _documents()
    doc = json.loads(encoded["components/first.json"])
    record = next(name for name in encoded if name.startswith("component_contracts/"))
    with pytest.raises(ValueError, match="record digest"):
        decode_contract_document(doc, lambda name: b"{}")
    # Both missing and duplicate fragments fail before component admission.
    for copies in (0, 2):
        archive_path = tmp_path / (str(copies) + ".tcm")
        with ZipFile(archive_path, "w") as archive:
            archive.writestr("components/first.json", encoded["components/first.json"])
            if copies:
                archive.writestr(record, encoded[record])
                with pytest.warns(UserWarning, match="Duplicate name"):
                    archive.writestr(record, encoded[record])
        with _document_reader(archive_path) as read, pytest.raises(ValueError, match="exactly once"):
            decode_contract_document(doc, read)


def test_shared_plan_calls_preserve_order_initialization_and_validation():
    rng = random.Random(23)
    # Long legal binding names represent wide captured ABIs whose call records
    # repeat across roots and beyond deflate's dictionary window.
    call = dict(kind="kernel", entry="compute", arguments=[rng.randbytes(256).hex() for _ in range(100)])
    plan = dict(version=2, operations=[deepcopy(call)] * 3, initialization=[deepcopy(call)])
    plans = {"executions/first.json": plan, "executions/second.json": deepcopy(plan)}
    encoded = encode_plan_documents(plans)
    assert len([n for n in encoded if n.startswith("execution_records/")]) == 1
    for name, expected in plans.items():
        document = json.loads(encoded[name])
        restored = decode_plan_document(document, encoded.__getitem__)
        assert restored == expected
        assert json.loads(ExecutionPlan.from_json(json.dumps(restored)).to_json()) == json.loads(
            ExecutionPlan.from_json(json.dumps(expected)).to_json())
        restored["operations"][0]["arguments"].append("changed")
        assert restored["initialization"][0] == call
        assert restored["operations"][1] == call
        for bad_path in (["version"], ["operations", -1], ["operations", True]):
            bad = deepcopy(document)
            bad["references"][0][0] = bad_path
            with pytest.raises(ValueError, match="record (path|reference)"):
                decode_plan_document(bad, encoded.__getitem__)
        bad = deepcopy(document)
        bad["plan"]["version"] = 1000
        with pytest.raises(ValueError, match="plan digest"):
            decode_plan_document(bad, encoded.__getitem__)
    invalid = {"executions/invalid.json": {**deepcopy(plan), "version": 1000}}
    stored = encode_plan_documents(invalid)
    restored = decode_plan_document(json.loads(stored["executions/invalid.json"]), stored.__getitem__)
    with pytest.raises(ValueError, match="Unsupported execution plan version"):
        ExecutionPlan.from_json(json.dumps(restored))


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_shared_contract_export_and_loaded_instances(tmp_path):
    env = Component(**_environment().declarations,
                    options=Record(scale=StaticValue(choices=(False, True)),
                                   capacity_stage=StaticValue(choices=(False, True))),
                    repeat=RootOption(choices=(1, 128)))
    model = _NestedConfiguration(7)
    root = env.compile_graph(model.record, repeat=128)
    module = ti.aot.Module()
    # Multiple public roots share a numerical implementation without sharing
    # component contracts or instance state.
    for index in range(4):
        root.export(module, "step" + str(index))
    legacy, shared = tmp_path / "legacy.tcm", tmp_path / "shared.tcm"
    module.archive(str(legacy))
    module.archive(str(shared), shared_component_contracts=True, shared_execution_plans=True)
    from taichi_forge.tools._component_archive import share_component_contracts

    rewritten = tmp_path / "rewritten.tcm"
    report = share_component_contracts(legacy, rewritten, shared_execution_plans=True)
    assert report["changed"] and report["after_bytes"] < report["before_bytes"]
    assert report["noncontract_compressed_byte_delta"] == 0
    with pytest.raises(ValueError, match="new destination"):
        share_component_contracts(legacy, legacy)
    with pytest.raises(ValueError, match="new destination"):
        share_component_contracts(legacy, rewritten)
    with ZipFile(legacy) as old, ZipFile(shared) as new:
        assert any(name.startswith("component_contracts/") for name in new.namelist())
        assert shared.stat().st_size < legacy.stat().st_size
        for name in old.namelist():
            if not name.startswith(("components/", "executions/")):
                assert new.read(name) == old.read(name)
            elif name.startswith("executions/"):
                assert decode_plan_document(json.loads(new.read(name)), new.read) == json.loads(old.read(name))
        directory = tmp_path / "directory"
        new.extractall(directory)
    for index in range(4):
        name = "step" + str(index)
        assert inspect_component(shared, name).contract == inspect_component(legacy, name).contract
        assert inspect_component(directory, name).contract == inspect_component(legacy, name).contract
        assert inspect_component(rewritten, name).contract == inspect_component(legacy, name).contract
    with Runtime("cpu") as runtime, runtime.load_module(shared) as loaded:
        component = loaded.component("step0")
        configuration = {name: (128 if source["path"] == ["repeat"] else
                                source["path"][-1] == "scale")
                         for name, source in component.configuration_sources().items()}
        with component.instantiate(dict(capacity=13, active=11), configuration=configuration) as first, \
                component.instantiate(dict(capacity=17, active=9), configuration=configuration) as second:
            for instance, gravity in ((first, 3), (second, 5)):
                instance.initialize(zero=component.contract["resources"])
                with instance.prepare(values=dict(gravity=[gravity, 0, 0])) as call:
                    call.run()
            np.testing.assert_array_equal(np.frombuffer(first.resource("x").read(), np.float32), [6]*11+[0]*2)
            np.testing.assert_array_equal(np.frombuffer(second.resource("x").read(), np.float32), [10]*9+[0]*8)
        with pytest.raises(ValueError, match="different compiled variant"):
            component.instantiate(dict(capacity=13, active=11), configuration={**configuration, "repeat": 1})
    # A fresh consumer uses only the Runtime and the artifact. The producer's
    # classes and numerical frontend cannot satisfy a missing storage dependency.
    probe = r'''
import importlib.abc, json, sys
sys.path.insert(0, sys.argv[1])
class NoFrontend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('taichi_forge', 'tests'):
            raise AssertionError('producer source imported')
sys.meta_path.insert(0, NoFrontend())
import numpy as np
from taichi_forge_runtime.aot import Runtime, inspect_component
info = inspect_component(sys.argv[2], 'step0')
config = json.loads(sys.argv[3])
info.check_configuration(config)
with Runtime('cpu') as runtime, runtime.load_module(sys.argv[2]) as module:
    component = module.component('step0')
    with component.instantiate(dict(capacity=13, active=11), configuration=config) as instance:
        instance.initialize(zero=component.contract['resources'])
        with instance.prepare(values=dict(gravity=[7, 0, 0])) as call:
            call.run()
        np.testing.assert_array_equal(np.frombuffer(instance.resource('x').read(), np.float32), [14]*11+[0]*2)
'''
    subprocess.run([sys.executable, "-c", probe, str(Path(taichi_forge_runtime.__file__).parent.parent),
                    str(rewritten), json.dumps(configuration)], check=True, capture_output=True, text=True)


@ti.kernel
def _stored_plan_increment(value: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    for i in range(value.shape[0]):
        value[i] += 1


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_shared_execution_records_load_directory_archive_and_frozen_rewrite(tmp_path):
    # A long diagnostic label is legal and deliberately exercises storage beyond
    # deflate's window without increasing kernel ABI or executing more kernels.
    label = random.Random(17).randbytes(20000).hex()
    call = dict(kind="kernel", entry="increment", arguments=["value"], label=label)
    plan = ExecutionPlan([call, deepcopy(call)])
    module = ti.aot.Module()
    module.add_kernel(_stored_plan_increment, template_args={"value": ti.ndarray(ti.i32, shape=(3,))}, name="increment")
    module.add_execution_plan("increment", plan)
    legacy, shared, rewritten = (tmp_path / name for name in ("legacy.tcm", "shared.tcm", "rewritten.tcm"))
    module.archive(str(legacy))
    module.archive(str(shared), shared_execution_plans=True)
    module.save(tmp_path / "directory", shared_execution_plans=True)
    from taichi_forge.tools._component_archive import share_component_contracts

    report = share_component_contracts(legacy, rewritten, shared_execution_plans=True)
    assert report["changed"] and report["execution_plan_count"] == 1
    assert report["noncontract_compressed_byte_delta"] == 0
    with ZipFile(shared) as archive:
        assert any(name.startswith("execution_records/") for name in archive.namelist())
    probe = r'''
import importlib.abc, json, struct, sys
from contextlib import closing
sys.path.insert(0, sys.argv[1])
class NoFrontend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('taichi_forge', 'tests'):
            raise AssertionError('producer source imported')
sys.meta_path.insert(0, NoFrontend())
from taichi_forge_runtime.aot import Runtime
for path in sys.argv[2:]:
    with Runtime('cpu') as runtime, runtime.load_module(path) as module:
        document = json.loads(module.execution_plan('increment').to_json())
        assert len(document['operations']) == 2
        with closing(runtime.ndarray('i32', (3,))) as first, closing(runtime.ndarray('i32', (5,))) as second:
            first.write(struct.pack('3i', 1, 2, 3))
            second.write(struct.pack('5i', 5, 6, 7, 8, 9))
            with module.prepare_execution('increment', {'value': first}) as a, module.prepare_execution('increment', {'value': second}) as b:
                a.run()
                b.run()
            assert struct.unpack('3i', first.read()) == (3, 4, 5)
            assert struct.unpack('5i', second.read()) == (7, 8, 9, 10, 11)
'''
    subprocess.run([sys.executable, "-c", probe, str(Path(taichi_forge_runtime.__file__).parent.parent),
                    str(shared), str(tmp_path / "directory"), str(rewritten)],
                   check=True, capture_output=True, text=True)
