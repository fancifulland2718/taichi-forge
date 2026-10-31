"""Cold artifact selection reuses loaded-component static admission rules."""
import json
import subprocess
import sys
from zipfile import ZipFile

import pytest
import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime, inspect_component
from tests import test_utils
from tests.python.test_aot_component_configuration import _ConfiguredModel, _configured


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_component_inspection_without_native_load(tmp_path):
    model = _ConfiguredModel(7)
    root = _configured().compile_graph(model.record)
    module = ti.aot.Module()
    root.export(module, "step")
    artifact = tmp_path / "model.tcm"
    module.archive(str(artifact))
    with ZipFile(artifact) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
        directory = tmp_path / "directory"
        archive.extractall(directory)
    configuration = dict(scale=True, extra_stage=False, unused_mode="a")
    info = inspect_component(artifact, "step")
    assert info.contract == inspect_component(directory, "step").contract
    assert info.semantic_key == root.contract["semantic_key"]
    assert info.target == "x64"
    info.check_configuration(configuration)
    copy = info.contract
    copy["configuration"].clear()
    sources = info.configuration_sources()
    sources["scale"]["path"].append("changed")
    assert info.configuration_sources()["scale"]["path"] == ["scale"]
    with Runtime("cpu") as runtime, runtime.load_module(artifact) as loaded:
        component = loaded.component("step")
        assert component.contract == info.contract
        assert component.configuration_sources() == info.configuration_sources()
        for supplied, message in (
            ({}, "configuration inputs differ"),
            ({**configuration, "scale": 1}, "outside its declared domain"),
            ({**configuration, "scale": False}, "different compiled variant"),
        ):
            with pytest.raises(ValueError, match=message):
                info.check_configuration(supplied)
            with pytest.raises(ValueError, match=message):
                component.instantiate(dict(capacity=13, active=11), configuration=supplied)
    # Import and query in a fresh process. Native loading and the numerical
    # frontend are forbidden; this does not borrow the producer's module state.
    probe = r'''
import ctypes, importlib.abc, sys
class NoFrontend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == 'taichi_forge' or fullname.startswith('taichi_forge.'):
            raise AssertionError('numerical frontend imported')
sys.meta_path.insert(0, NoFrontend())
def no_native(*args, **kwargs):
    raise AssertionError('native library loaded')
ctypes.CDLL = no_native
if hasattr(ctypes, 'WinDLL'):
    ctypes.WinDLL = no_native
from taichi_forge_runtime.aot import inspect_component
info = inspect_component(sys.argv[1], 'step')
info.check_configuration(dict(scale=True, extra_stage=False, unused_mode='a'))
assert info.configuration_sources()['scale']['path'] == ['scale']
'''
    subprocess.run([sys.executable, "-c", probe, str(artifact)], check=True, capture_output=True, text=True)

    def changed(label, mutate):
        contract = json.loads(contents["components/step.json"])
        mutate(contract)
        path = tmp_path / (label + ".tcm")
        with ZipFile(path, "w") as archive:
            for name, payload in contents.items():
                archive.writestr(name, json.dumps(contract) if name == "components/step.json" else payload)
        return path

    for label, mutate, message in (
        ("version", lambda c: c.update(version=999), "Unsupported component"),
        ("source", lambda c: c["compilation"]["configuration_sources"]["scale"].update(path=["../scale"]),
         "source path"),
        ("static", lambda c: c["configuration"]["scale"].update(selected=["bool", False]),
         "differs from compiled static dependencies"),
    ):
        with pytest.raises(ValueError, match=message):
            inspect_component(changed(label, mutate), "step")
    legacy = changed("legacy", lambda c: c["compilation"].pop("configuration_sources"))
    with pytest.raises(ValueError, match="rebuild it"):
        inspect_component(legacy, "step").configuration_sources()
    duplicate = tmp_path / "duplicate.tcm"
    with ZipFile(duplicate, "w") as archive:
        archive.writestr("components/step.json", contents["components/step.json"])
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("components/step.json", contents["components/step.json"])
    with pytest.raises(ValueError, match="exactly once"):
        inspect_component(duplicate, "step")
    # Inspection is not full resource/device admission. Keep the loaded
    # component's independent capability rejection intact.
    unsupported = changed("capability", lambda c: c["required_capabilities"].append("future_capability"))
    inspect_component(unsupported, "step").check_configuration(configuration)
    with Runtime("cpu") as runtime, runtime.load_module(unsupported) as loaded:
        with pytest.raises(ValueError, match="capabilit"):
            loaded.component("step")
