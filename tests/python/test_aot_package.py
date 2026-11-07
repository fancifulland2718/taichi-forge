"""Definition coverage and source-free consumption of the initial package slice."""

import ast
import importlib
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge._lib import utils
from tests import test_utils
from taichi_forge_runtime.aot import Runtime, load_package

SOURCE = """
import taichi_forge as ti
from .device_helpers import scale
constructors = 0

@ti.data_oriented
class Model:
    def __init__(self, n: int, gain: float):
        global constructors
        constructors += 1
        self.n = n
        self.gain = gain
        self.values = ti.ndarray(ti.f32, shape=n)
        self.initialize(self.values, 0.5)

    @ti.func
    def affine(self, x):
        return scale(x, self.gain)

    @ti.kernel
    def initialize(self, values: ti.types.ndarray(dtype=ti.f32, ndim=1), offset: ti.f32):
        for i in range(self.n):
            values[i] = self.affine(i) + offset

    @ti.kernel
    def add(self, values: ti.types.ndarray(dtype=ti.f32, ndim=1), amount: ti.f32):
        for i in values:
            values[i] += amount

    @ti.kernel
    def subtract(self, values: ti.types.ndarray(dtype=ti.f32, ndim=1), amount: ti.f32):
        for i in values:
            values[i] -= amount

    def step(self, positive, amount):
        if positive:
            self.add(self.values, amount)
        else:
            self.subtract(self.values, amount)

    def description(self):
        # Same name as a numerical method, but this is ordinary local Host code.
        def initialize():
            return f"Model with {self.n} values"
        return initialize()

@ti.kernel
def unused_public_function(x: ti.i32) -> ti.i32:
    return x * 3 + 7
"""


def _source_package(tmp_path, monkeypatch, source=SOURCE):
    root = tmp_path / "definition_fixture"
    root.mkdir()
    (root / "__init__.py").write_text("from .model import Model\n", encoding="utf8")
    (root / "model.py").write_text(textwrap.dedent(source), encoding="utf8")
    (root / "device_helpers.py").write_text(
        textwrap.dedent("""
        from taichi_forge import func as device
        @device
        def scale(x, gain):
            return x * gain
    """),
        encoding="utf8",
    )
    (root / "unimported.py").write_text(
        textwrap.dedent("""
        from taichi_forge import kernel as device, i32 as integer
        @device
        def available_without_a_test(x: integer) -> integer:
            return x * 5
    """),
        encoding="utf8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    for key in (
        "definition_fixture",
        "definition_fixture.model",
        "definition_fixture.unimported",
        "definition_fixture.device_helpers",
    ):
        monkeypatch.delitem(sys.modules, key, raising=False)
    return importlib.import_module("definition_fixture")


@pytest.mark.parametrize("helper_kind", ["func", "real_func"])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_compiles_definitions_without_constructor_or_test_trace(tmp_path, monkeypatch, helper_kind):
    source = SOURCE.replace(
        "@ti.func\n    def affine(self, x):", f"@ti.{helper_kind}\n    def affine(self, x: ti.f32) -> ti.f32:"
    )
    package = _source_package(tmp_path, monkeypatch, source)
    assert "definition_fixture.unimported" not in sys.modules
    model_module = sys.modules["definition_fixture.model"]
    result = ti.aot.export_package(package, tmp_path / "sdk", arch=ti.cfg.arch)
    assert model_module.constructors == 0
    assert {item["definition"].rsplit(".", 1)[-1] for item in result["entries"].values()} == {
        "initialize",
        "add",
        "subtract",
        "unused_public_function",
        "available_without_a_test",
    }
    host = (tmp_path / "sdk/host/model.py").read_text(encoding="utf8")
    assert "scale" not in host
    assert "return x * gain" not in (tmp_path / "sdk/host/device_helpers.py").read_text(encoding="utf8")
    assert not any(
        isinstance(n, ast.FunctionDef) and n.name in {"affine", "initialize", "add", "subtract"}
        for cls in ast.parse(host).body
        if isinstance(cls, ast.ClassDef)
        for n in cls.body
    )
    assert "def step" in host and "def __init__" in host and "def description" in host
    assert not list((tmp_path / "sdk").rglob("*.pyd"))

    # Ordinary JIT continues using the unmodified source class and decorators.
    jit = package.Model(7, 2.0)
    jit.step(False, 1.25)
    np.testing.assert_array_equal(jit.values.to_numpy(), np.arange(7, dtype=np.float32) * 2 - 0.75)

    probe = tmp_path / "consumer.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib
        import importlib.abc
        import json
        import sys
        import numpy as np
        class NoFrontend(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError('Private frontend/source import: ' + fullname)
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime:
            with load_package(runtime, sys.argv[1]) as code:
                a, b = code.host.Model(7, 2.0), code.host.Model(19, -3.0)
                a.step(False, 1.25)
                b.step(True, 4.0)
                read = lambda model: np.frombuffer(model.values.read(), np.float32)
                np.testing.assert_array_equal(read(a), np.arange(7, dtype=np.float32) * 2 - .75)
                np.testing.assert_array_equal(read(b), np.arange(19, dtype=np.float32) * -3 + 4.5)
                assert a.description() == 'Model with 7 values'
                module = importlib.import_module(code.host.__name__ + '.model')
                assert module.unused_public_function(5) == 22
                with load_package(runtime, sys.argv[1]) as other:
                    c = other.host.Model(3, 4.0)
                    late = importlib.import_module(code.host.__name__ + '.unimported')
                    assert late.available_without_a_test(4) == 20
                    a.gain = 5.0
                    a.initialize(a.values, offset=2.0)  # bound to a's package, not the active other package
                    np.testing.assert_array_equal(read(a), np.arange(7, dtype=np.float32) * 5 + 2)
                    np.testing.assert_array_equal(read(c), np.arange(3, dtype=np.float32) * 4 + .5)
                assert late.available_without_a_test(7) == 35
                b.step(False, 1.0)
                np.testing.assert_array_equal(read(b), np.arange(19, dtype=np.float32) * -3 + 3.5)
            try:
                a.add(a.values, 1.0)
            except RuntimeError as error:
                assert 'closed' in str(error)
            else:
                raise AssertionError('Retired package accepted invocation')
        assert 'taichi_forge' not in sys.modules and 'taichi' not in sys.modules
        print(json.dumps({'source_free': True, 'instances': 3, 'both_host_branches': True}))
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), result["target"]],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"source_free": true' in result.stdout


@pytest.mark.parametrize(
    "source, diagnostic",
    [
        (
            SOURCE.replace(
                "self.values = ti.ndarray(ti.f32, shape=n)",
                "if n > 0:\n            self.values = ti.ndarray(ti.f32, shape=n)",
            ),
            "conditional",
        ),
        (SOURCE.replace("ti.ndarray(ti.f32, shape=n)", "ti.field(ti.f32, shape=n)"), "Host frontend operation 'field'"),
        (SOURCE.replace("range(self.n)", "ti.static(range(self.n))"), "runtime capture"),
        (
            SOURCE.replace('return f"Model with {self.n} values"', "return self.affine(3)"),
            "Host references a device-only helper",
        ),
        (
            SOURCE.replace('return f"Model with {self.n} values"', "return scale(3, self.gain)"),
            "Host references a device-only helper",
        ),
        (
            SOURCE.replace('return f"Model with {self.n} values"', "return ti.f32(self.gain)"),
            "Host scalar casts",
        ),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_unproven_definitions_without_publishing(tmp_path, monkeypatch, source, diagnostic):
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()
    assert sys.modules["definition_fixture.model"].constructors == 0


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_backend_selection_keeps_live_jit_and_initializes_cold_backend(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch)
    program = ti.lang.impl.get_runtime().prog
    with pytest.raises(ValueError, match="live JIT"):
        ti.aot.export_package(package, tmp_path / "wrong", arch="cuda")
    assert ti.lang.impl.get_runtime().prog is program
    with pytest.raises(ValueError, match="arch must"):
        ti.aot.export_package(package, tmp_path / "unknown", arch="automatic")
    ti.reset()
    result = ti.aot.export_package(package, tmp_path / "cold", arch="cpu")
    assert result["target"] == ti.cpu.name
    assert ti.lang.impl.current_cfg().arch == ti.cpu


COMPOUND_SOURCE = """
import taichi_forge as ti
Vec = ti.types.vector(3, ti.f32)
Mat = ti.types.matrix(2, 2, ti.f32)
@ti.dataclass
class Record:
    value: ti.f32
    vector: Vec
    matrix: Mat

    @ti.func
    def total(self):
        return self.value + self.vector[2] + self.matrix[1, 1]

@ti.data_oriented
class Model:
    def __init__(self, n: int):
        self.records = ti.ndarray(Record, shape=n)
        self.vectors = ti.ndarray(Vec, shape=n)
        self.matrices = ti.ndarray(Mat, shape=n)
        self.output = ti.ndarray(ti.f32, shape=n)

    @ti.kernel
    def advance(self, records: ti.types.ndarray(Record, 1), vectors: ti.types.ndarray(Vec, 1),
                matrices: ti.types.ndarray(Mat, 1), output: ti.types.ndarray(ti.f32, 1),
                vector: Vec, matrix: Mat, scalar_record: Record):
        for i in output:
            records[i].value = i + scalar_record.value
            records[i].vector = vector
            records[i].matrix = matrix
            vectors[i] = vector * 2
            matrices[i] = matrix * 3
            output[i] = records[i].total() + scalar_record.vector[0]

    def step(self):
        self.advance(self.records, self.vectors, self.matrices, self.output,
                     Vec(2, 3, 4), Mat([[1, 2], [3, 5]]),
                     Record(value=7, vector=Vec(11, 13, 17), matrix=Mat(0)))
"""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_uses_native_struct_layout_and_compound_values(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, COMPOUND_SOURCE)
    result = ti.aot.export_package(package, tmp_path / "sdk", arch=ti.cfg.arch.name)
    ordinary = package.Model(7)
    ordinary.step()
    expected = ordinary.output.to_numpy()
    with Runtime(result["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as library:
        a, b = library.host.Model(7), library.host.Model(19)
        a.step()
        b.step()
        np.testing.assert_array_equal(np.frombuffer(a.output.read(), np.float32), expected)
        np.testing.assert_array_equal(np.frombuffer(b.output.read(), np.float32), np.arange(19) + 27)
        assert a.records.dtype.element_stride == b.records.dtype.element_stride
        np.testing.assert_array_equal(np.frombuffer(b.vectors.read(), np.float32).reshape(19, 3), [[4, 6, 8]] * 19)
        np.testing.assert_array_equal(
            np.frombuffer(b.matrices.read(), np.float32).reshape(19, 2, 2), [[[3, 6], [9, 15]]] * 19
        )


REAL_SOURCE = """
import taichi_forge as ti

@ti.real_func
def sum_to(n: ti.i32) -> ti.i32:
    result = 0
    if n > 0:
        result = n + sum_to(n - 1)
    return result

@ti.kernel
def sum_kernel(n: ti.i32) -> ti.i32:
    return sum_to(n)

@ti.data_oriented
class Model:
    def __init__(self, n: int, gain: float):
        self.gain = gain
        self.values = ti.ndarray(ti.f32, shape=n)

    @ti.real_func
    def write(self, i: ti.i32):
        self.values[i] = sum_to(i) * self.gain

    @ti.real_func
    def nested(self, i: ti.i32):
        self.write(i)

    @ti.kernel
    def run(self):
        for i in range(self.values.shape[0]):
            self.nested(i)
"""


@pytest.mark.parametrize("recursive", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_package_real_functions_keep_nested_array_captures_and_explicit_recursion(tmp_path, monkeypatch, recursive):
    source = (
        REAL_SOURCE if recursive else REAL_SOURCE.replace("result = n + sum_to(n - 1)", "result = n * (n + 1) // 2")
    )
    # Keep device recursion shallow: standalone Runtime currently has no CUDA
    # stack-limit setting. Sizes still vary independently of the call depth.
    source = source.replace("sum_to(i) * self.gain", "sum_to(i % 3) * self.gain")
    package = _source_package(tmp_path, monkeypatch, source)
    result = ti.aot.export_package(package, tmp_path / "sdk")
    # Ordinary JIT requires ndarrays to be explicit arguments. Deployment
    # capture supplies those hidden arguments; it does not alter that JIT rule.
    ordinary = sys.modules["definition_fixture.model"].sum_kernel
    assert ordinary(6) == 21
    with Runtime(result["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as library:
        a, b = library.host.Model(7, 2.0), library.host.Model(19, -3.0)
        a.run()
        b.run()
        small, large = np.arange(7) % 3, np.arange(19) % 3
        np.testing.assert_array_equal(np.frombuffer(a.values.read(), np.float32), small * (small + 1))
        np.testing.assert_array_equal(np.frombuffer(b.values.read(), np.float32), large * (large + 1) * -1.5)
        a.gain = 4.0
        a.run()
        np.testing.assert_array_equal(np.frombuffer(a.values.read(), np.float32), small * (small + 1) * 2)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_recursive_real_function_hidden_captures(tmp_path, monkeypatch):
    source = REAL_SOURCE.replace(
        "self.values[i] = sum_to(i) * self.gain",
        "if i > 0:\n            self.write(i - 1)\n        self.values[i] = self.gain",
    )
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="Recursive real_func captures"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_package_diagnoses_real_function_backend_before_publication(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, REAL_SOURCE)
    with pytest.raises(ti.aot.PackageExportError, match="real_func AOT requires an LLVM backend"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()
