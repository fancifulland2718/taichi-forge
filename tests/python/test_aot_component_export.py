"""A frozen original Field entry consumed with only the Runtime package."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile

import taichi_forge as ti
import taichi_forge_runtime
from tests import test_utils
from tests.python.test_aot_component_dense import _DenseModel, _environment


_CONSUMER = r"""
import importlib.abc
import json
from pathlib import Path
import struct
import sys

class NoNumericalSource(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('taichi_forge', 'tests', 'geophys'):
            raise ImportError('Private numerical sources are unavailable')

sys.meta_path.insert(0, NoNumericalSource())
sys.path.insert(0, sys.argv[1])
from taichi_forge_runtime.aot import Runtime

with Runtime(sys.argv[3]) as runtime:
    assert Path(runtime._api.library._name).resolve().is_relative_to(Path(sys.argv[1]).resolve())
    module = runtime.load_module(sys.argv[2])
    active = module.component_entry_contract('active')
    capacity = module.component_entry_contract('capacity')
    contract = active['contract']
    results = []
    # Fresh host inputs, created after the TCM was frozen. There is no producer
    # object, field allocation code or numerical function in this process.
    for n, count, gravity in ((37, 31, [5, 7, 9]), (129, 0, [2, 3, 4])):
        dimensions = dict(capacity=n, active=count)
        assert 0 <= count <= n
        groups = {}
        for name, schema_json in contract['groups'].items():
            schema = json.loads(schema_json)
            required = {e['parameter'] for node in schema['nodes'] for e in node['extents'] if e['parameter']}
            groups[name] = runtime.dense_snode_group(schema_json, {d: dimensions[d] for d in required})
        views = {}

        def bind(projection):
            arguments = []
            for name, declaration in projection['captures']:
                kind = declaration[0]
                if kind == 'dimension':
                    arguments.append(dimensions[name])
                elif kind == 'value':
                    assert name == 'gravity'
                    arguments.append(gravity)
                else:
                    assert kind == 'dense_field'
                    group, paths = projection['members'][name]
                    dtype, element_shape = declaration[1][1:]
                    key = (group, json.dumps(paths))
                    if key not in views:
                        views[key] = groups[group].member(paths, dtype, element_shape=element_shape)
                    arguments.append(views[key])
            return arguments

        args = bind(contract)
        arrays = {name: value for (name, decl), value in zip(contract['captures'], args) if decl[0] == 'dense_field'}
        sentinel = dict(alias_x=-11, x=-11, y=-17, v=-23, m=-29)
        for name, array in arrays.items():
            array.write(struct.pack('<f', sentinel[name]) * (array.nbytes // 4))
        with module.kernel('active').prepare(*args) as prepared:
            prepared.run()
        x = arrays.get('x', arrays.get('alias_x'))
        values = struct.unpack('<' + 'f' * n, x.read())
        assert values == tuple([-11 + gravity[0]] * count + [-11] * (n-count))
        assert struct.unpack('<' + 'f' * n, arrays['y'].read()) == tuple(
            [2 * (-11 + gravity[0])] * count + [-17] * (n-count))
        assert all(value == -29 for value in struct.unpack('<' + 'f' * (4*n), arrays['m'].read())[4*count:])
        cap_args = bind(capacity['contract'])
        with module.kernel('capacity').prepare(*cap_args) as prepared:
            prepared.run()
        assert struct.unpack('<' + 'f' * n, arrays['y'].read()) == tuple(range(n))
        assert struct.unpack('<' + 'f' * (3*n), arrays['v'].read())[-3:] == (n-1, n, n+1)
        results.append(dict(capacity=n, active=count, tail=values[-1]))
        for group in groups.values():
            group.close()
        for array in views.values():
            array.close()
assert not any(name == 'taichi_forge' or name.startswith('taichi_forge.') for name in sys.modules)
print(json.dumps(dict(results=results, private_numerical_source=False)))
"""


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_original_field_entries_export_and_consume_without_numerical_source(tmp_path):
    producer = _DenseModel(7, 7, gravity=(101, 103, 107))
    env = _environment()
    active = env.compile(producer.active_pass)
    capacity = env.compile(producer.capacity_pass)
    module = ti.aot.Module()
    active.export(module, "active")
    active.export(module, "active")  # Idempotent publication of the same contract.
    capacity.export(module, "capacity")
    artifact = tmp_path / "component.tcm"
    module.archive(str(artifact))
    before = hashlib.sha256(artifact.read_bytes()).hexdigest()
    with ZipFile(artifact) as archive:
        assert not any(name.startswith("data/") for name in archive.namelist())
        projected = json.loads(archive.read("component_entries/active.json"))
        assert projected["semantic_key"] == active.semantic_key
        assert projected["initialization"]["model_data"] == "consumer"
        assert not module._aot_builder._used_snode_tree_ids(["active", "capacity"])

    # Only Runtime sources/native libraries are copied. No DSL or model source
    # is placed in the consumer directory; -I also ignores the producer path.
    package_root = tmp_path / "consumer"
    shutil.copytree(
        Path(taichi_forge_runtime.__file__).parent,
        package_root / "taichi_forge_runtime",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            _CONSUMER,
            str(package_root),
            str(artifact),
            "cpu" if ti.cfg.arch == ti.cpu else "cuda",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert not report["private_numerical_source"]
    assert [item["capacity"] for item in report["results"]] == [37, 129]
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == before
