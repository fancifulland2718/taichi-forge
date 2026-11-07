"""Definition coverage and source-free consumption of the initial package slice."""

import ast
import importlib
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import zipfile

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


def _source_package(tmp_path, monkeypatch, source=SOURCE, *, initializer="from .model import Model\n"):
    root = tmp_path / "definition_fixture"
    root.mkdir()
    (root / "__init__.py").write_text(initializer, encoding="utf8")
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


@pytest.mark.parametrize("helper_kind", ["func", "real_func", "pyfunc"])
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
    expected_entries = {
        "initialize",
        "add",
        "subtract",
        "unused_public_function",
        "available_without_a_test",
    }
    if helper_kind == "pyfunc":
        expected_entries.add("affine")
    assert {item["definition"].rsplit(".", 1)[-1] for item in result["entries"].values()} == expected_entries
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
                if sys.argv[3] == 'pyfunc':
                    assert a.affine(3.0) == 6.0 and b.affine(3.0) == -9.0
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
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), result["target"], helper_kind],
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
        (SOURCE.replace("ti.ndarray(ti.f32, shape=n)", "ti.field(ti.f32, shape=n, order='i')"), "plain default AOS"),
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


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_preserves_host_ndarray_annotations_without_frontend(tmp_path, monkeypatch):
    source = COMPOUND_SOURCE
    aliases = {}
    for element in ("Record", "Vec", "Mat", "ti.f32"):
        alias = element.replace(".", "_") + "Array"
        annotation = f"ti.types.ndarray({element}, 1)"
        source = source.replace(annotation, alias)
        aliases[alias] = annotation
    source = source.replace(
        "@ti.data_oriented", "\n".join(f"{k} = {v}" for k, v in aliases.items()) + "\n@ti.data_oriented"
    )
    source += "\ndef accept_array(value: VecArray) -> VecArray:\n    return value\n"
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 9  # Host tensor type factories need only logical Runtime metadata.
    original = package.Model(7)
    original.step()
    np.testing.assert_array_equal(original.output.to_numpy(), np.arange(7) + 27)
    script = tmp_path / "consume_annotations.py"
    script.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class NoFrontend(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            module = code.host.model
            assert module.VecArray.ndim == 1
            assert module.VecArray.dtype.spec['element_shape'] == [3]
            assert module.ti_f32Array.dtype == 'f32'
            assert module.VecArray.boundary == 'UNSAFE'
            assert module.accept_array.__annotations__['value'] is module.VecArray
            for n in (7, 19):
                model = code.host.Model(n)
                model.step()
                assert module.accept_array(model.vectors) is model.vectors
                np.testing.assert_array_equal(np.frombuffer(model.output.read(), np.float32), np.arange(n) + 27)
                np.testing.assert_array_equal(np.frombuffer(model.vectors.read(), np.float32).reshape(n, 3), [[4, 6, 8]] * n)
            assert not any(name.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture') for name in sys.modules)
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(script), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_incomplete_host_ndarray_annotations(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, SOURCE + "\nUnknown = ti.types.ndarray(ti.f32)\n")
    with pytest.raises(ti.aot.PackageExportError, match="fixed rank"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


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


PYFUNC_SOURCE = """
import taichi_forge as ti

@ti.pyfunc
def twice(x: ti.i32) -> ti.i32:
    return x * 2

@ti.data_oriented
class Model:
    def __init__(self, bias: int):
        self.bias = bias

    @ti.pyfunc
    def adjust(self, x: ti.i32) -> ti.i32:
        return twice(x) + self.bias

    @ti.kernel
    def run(self, x: ti.i32) -> ti.i32:
        return self.adjust(x)

    def host(self, x):
        return self.adjust(x) + twice(x + 1)
"""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_typed_pyfunc_is_available_from_host_and_device(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, PYFUNC_SOURCE)
    original = package.Model(3)
    expected = original.host(5)
    assert original.run(5) == 13
    result = ti.aot.export_package(package, tmp_path / "sdk")
    projected = (tmp_path / "sdk/host/model.py").read_text(encoding="utf8")
    assert "return x * 2" not in projected and "return twice(x) + self.bias" not in projected
    with Runtime(result["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as library:
        a, b = library.host.Model(3), library.host.Model(-7)
        assert a.host(5) == expected and a.run(5) == 13
        assert b.host(5) == expected - 10 and b.run(5) == 3
        a.bias = 9
        assert a.host(5) == expected + 6


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_pyfunc_without_declared_domain_does_not_sample_calls(tmp_path, monkeypatch):
    source = PYFUNC_SOURCE.replace("def twice(x: ti.i32)", "def twice(x)")
    package = _source_package(tmp_path, monkeypatch, source)
    # Calling it before export does not establish a production signature.
    assert package.Model(3).host(5) == 25
    with pytest.raises(ti.aot.PackageExportError, match="pyfunc export requires explicit parameter types"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_scope_dependent_pyfunc(tmp_path, monkeypatch):
    source = PYFUNC_SOURCE.replace("return x * 2", "return x * 2 if ti.lang.impl.inside_kernel() else x * 3")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="scope-dependent pyfunc"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_submodule_device_imports_and_host_graph_share_code(tmp_path, monkeypatch):
    package = _source_package(
        tmp_path,
        monkeypatch,
        SOURCE + """

def record(builder):
    builder.dispatch(unused_public_function, ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'x', ti.i32))
""",
    )
    module = sys.modules["definition_fixture.model"]
    # Select one source module, not the parent package: its relative helper is
    # compiler-only and must not leak a dependency on the private parent.
    result = ti.aot.export_package(module, tmp_path / "sdk")
    assert len(result["entries"]) == 4
    assert ".device_helpers" not in (tmp_path / "sdk/host/__init__.py").read_text(encoding="utf8")
    with Runtime(result["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as library:
        builder = library.host.ti.graph.GraphBuilder()
        library.host.record(builder)
        graph = builder.compile()
        with graph.bind(dict(x=4)) as binding:
            binding.run()
        with pytest.raises(ValueError, match="differs from the entry's type"):
            builder.dispatch(
                library.host.unused_public_function,
                library.host.ti.graph.Arg(library.host.ti.graph.ArgKind.SCALAR, "wrong", "f32"),
            )
        # Host recording references the same compiled definition; it never
        # creates another artifact for this sequence or argument value.
        assert len(library.manifest["entries"]) == 4


OWNER_GRAPH_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, gain: float):
        global constructors
        constructors += 1
        self.n = n
        self.gain = gain
        self.bias = ti.Vector([1.0, 2.0], ti.f32)
        scope = ti.experimental.ResourceScope(self)
        scope.value('bias', ti.types.vector(2, ti.f32))
        scope.values = scope.field(ti.f32, shape=scope.dim('n'))

    @ti.kernel
    def reset(self):
        for i in self.values:
            self.values[i] = 0

    @ti.kernel
    def step(self, increment: ti.i32, extra: ti.types.vector(2, ti.f32)):
        for i in range(self.n):
            self.values[i] += self.gain * (i + increment) + self.bias[0] + extra[1]

    def record(self, builder):
        def vector(name, dtype):
            return ti.graph.Arg(ti.graph.ArgKind.MATRIX, name, ti.types.vector(2, dtype))
        extra = vector('extra', ti.f32)
        one = builder.private_scalar('one', ti.i32, 1)
        builder.dispatch(self.step, one, extra, template_args={'self': self})
        child = builder.create_sequential()
        child.dispatch(self.step, ti.graph.Arg(ti.graph.ArgKind.SCALAR, 'increment', ti.i32), extra,
                       template_args={'self': self})
        builder.append(child)

    def graph(self):
        builder = ti.graph.GraphBuilder()
        builder.dispatch(self.reset, template_args={'self': self})
        self.record(builder)
        return builder.compile()

    def compile_component(self):
        return ti.aot.Component.from_owner(self).compile_graph(self.record)

    def wait(self):
        import taichi_forge as ti
        assert ti.lang.impl.get_runtime().prog is not None
        ti.sync()
"""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_graph_captures_are_instance_snapshots_without_new_code(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, OWNER_GRAPH_SOURCE)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 9
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert [item["operation"] for item in manifest["producer_only"]] == ["aot.Component"]
    reference = package.Model(7, 2.0)
    reference.reset()
    for increment in (1, 3):
        reference.step(increment, ti.Vector([0.0, 0.0]))
    np.testing.assert_array_equal(reference.values.to_numpy(), np.arange(7) * 4 + 10)
    probe = tmp_path / "consume_owner_graph.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import hashlib
        import sys
        from pathlib import Path
        import numpy as np
        class NoFrontend(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime.aot import Runtime, load_package
        root = Path(sys.argv[1])
        before = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, root) as code:
            a, b = code.host.Model(7, 2.0), code.host.Model(19, -1.0)
            ga, gb = a.graph(), b.graph()
            assert len(ga.plan.argument_names) > len(ga.arguments)  # private captures, not public arguments
            assert ga.arguments.keys() == {'increment', 'extra'}
            pa = ga.bind(dict(increment=3, extra=(0.0, 0.0)))
            pb = gb.bind(dict(increment=5, extra=(0.0, 0.0)))
            a.gain = 9.0
            a.bias.x = 7.0
            pb.run()
            pa.run()
            pb.run()
            a.wait()
            np.testing.assert_array_equal(np.frombuffer(a.values.read(), np.float32), np.arange(7) * 4 + 10)
            np.testing.assert_array_equal(np.frombuffer(b.values.read(), np.float32), np.arange(19) * -2 - 4)
            with ga.bind(dict(increment=3, extra=(0.0, 0.0))) as fresh:
                fresh.run()
            np.testing.assert_array_equal(np.frombuffer(a.values.read(), np.float32), np.arange(7) * 18 + 50)
            pa.run()  # the earlier snapshot remains independent
            np.testing.assert_array_equal(np.frombuffer(a.values.read(), np.float32), np.arange(7) * 4 + 10)
            pa.close()
            pb.close()
            # Transfers use the native member view, including strided Host input.
            source = np.arange(14, dtype=np.float64)[::2]
            a.values.from_numpy(source)
            copy = a.values.to_numpy()
            np.testing.assert_array_equal(copy, source)
            copy[:] = -1
            np.testing.assert_array_equal(a.values.to_numpy(), source)
            try:
                a.values.from_numpy(np.zeros((7, 1), np.float32))
            except ValueError as error:
                assert 'shape' in str(error)
            else:
                raise AssertionError('Wrong transfer shape accepted')
            try:
                a.compile_component()
            except RuntimeError as error:
                assert 'producer-only' in str(error)
            else:
                raise AssertionError('Installed package attempted producer compilation')
            builder = code.host.model.ti.graph.GraphBuilder()
            builder.private_scalar('one', 'i32', 1)
            try:
                builder.private_scalar('one', 'i32', 2)
            except ValueError:
                pass
            else:
                raise AssertionError('Repeated private scalar accepted')
            with load_package(runtime, root) as other:
                foreign = other.host.Model(3, 1.0)
                try:
                    builder.dispatch(foreign.reset)
                except ValueError as error:
                    assert 'same loaded package' in str(error)
                else:
                    raise AssertionError('Foreign code owner accepted')
        after = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
        assert before == after
        print('OWNER_GRAPH_SOURCE_FREE_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OWNER_GRAPH_SOURCE_FREE_OK" in result.stdout


FIELD_SOURCE = """
import taichi_forge as ti
constructors = 0

@ti.data_oriented
class Model:
    def __init__(self, n: int, m: int):
        global constructors
        constructors += 1
        self.n = n
        self.a = ti.field(ti.f32, shape=(self.n, m))
        self.v = ti.Vector.field(3, ti.f32, shape=n)
        self.mat = ti.Matrix.field(2, 2, ti.f32, shape=m)
        self.total = ti.field(ti.f32, shape=())

    @ti.kernel
    def step(self, gain: ti.f32):
        for i, j in self.a:
            self.a[i, j] += gain * (i * 10 + j)
        for i in self.v:
            self.v[i] += ti.Vector([gain, i, self.a.shape[1]])
        for j in self.mat:
            self.mat[j] += ti.Matrix([[gain, j], [2, 3]])
        self.total[None] += gain * self.n
"""


@pytest.mark.parametrize("real_helper", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_dense_fields_use_native_group_layout_and_dynamic_instances(tmp_path, monkeypatch, real_helper):
    from tests.python.test_aot_dense_snode_group import _compiled_nodes

    if real_helper and ti.cfg.arch == ti.vulkan:
        pytest.skip("Vulkan does not support real functions")
    source = FIELD_SOURCE
    if real_helper:
        source = source.replace("self.a[i, j] += gain * (i * 10 + j)", "self.touch(i, j, gain)")
        source = source.replace(
            "    @ti.kernel\n    def step",
            """    @ti.real_func
    def touch(self, i: ti.i32, j: ti.i32, gain: ti.f32):
        self.a[i, j] += gain * (i * 10 + j)

    @ti.kernel
    def step""",
        )
    package = _source_package(tmp_path, monkeypatch, source)
    original = package.Model(7, 3)
    original.step(2.0)
    expected = {name: getattr(original, name).to_numpy() for name in ("a", "v", "mat", "total")}
    root = ti.lang.impl.get_runtime().prog.get_snode_root(original.a.snode.ptr.get_snode_tree_id())
    expected_layout = _compiled_nodes(root)
    result = ti.aot.export_package(package, tmp_path / "sdk")
    original.step(0.0)  # Existing JIT remains callable after structural export.
    np.testing.assert_array_equal(original.a.to_numpy(), expected["a"])
    assert sys.modules["definition_fixture.model"].constructors == 1  # no export-time model
    assert len(result["field_groups"]) == 1
    with Runtime(result["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as library:
        a, b = library.host.Model(7, 3), library.host.Model(19, 5)
        group, dimensions = next(iter(a._forge_package_field_groups.values()))
        assert group.layout["nodes"] == expected_layout
        assert dimensions == {
            "_forge_package_extent_arg_n": 7,
            "_forge_package_extent_arg_m": 3,
            "_forge_package_extent_self_n": 7,
        }
        assert a.a._group_owner is a.v._group_owner is a.mat._group_owner is a.total._group_owner
        assert a.a._group_owner is not b.a._group_owner
        assert not np.frombuffer(a.a.read(), np.float32).any()
        a.step(2.0)
        b.step(-3.0)
        for name, data in expected.items():
            actual = np.frombuffer(getattr(a, name).read(), np.float32).reshape(data.shape)
            np.testing.assert_array_equal(actual, data)
        np.testing.assert_array_equal(
            np.frombuffer(b.a.read(), np.float32).reshape(19, 5), -3 * (np.arange(19)[:, None] * 10 + np.arange(5))
        )
        assert np.frombuffer(b.total.read(), np.float32).item() == -57
        b.step(1.0)
        np.testing.assert_array_equal(np.frombuffer(a.a.read(), np.float32).reshape(7, 3), expected["a"])
        registered = set(runtime._resources)
        with pytest.raises((ValueError, RuntimeError)):
            library.host.Model(0, 3)
        assert runtime._resources == registered
    if not real_helper:
        probe = tmp_path / "consume_fields.py"
        probe.write_text(
            textwrap.dedent("""
            import importlib.abc
            import sys
            import numpy as np
            class NoFrontend(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                        raise AssertionError('Private frontend/source import: ' + fullname)
            sys.meta_path.insert(0, NoFrontend())
            from taichi_forge_runtime.aot import Runtime, load_package
            with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as library:
                model = library.host.Model(3, 2)
                model.step(2.0)
                np.testing.assert_array_equal(np.frombuffer(model.a.read(), np.float32), [0, 2, 20, 22, 40, 42])
                assert model.a._group_owner is model.v._group_owner is model.mat._group_owner
                assert np.frombuffer(model.total.read(), np.float32).item() == 6
            print('FIELD_PACKAGE_SOURCE_FREE_OK')
        """),
            encoding="utf8",
        )
        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
            TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
            TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        )
        consumer = subprocess.run(
            [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), result["target"]],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert consumer.returncode == 0, consumer.stdout + consumer.stderr
        assert "FIELD_PACKAGE_SOURCE_FREE_OK" in consumer.stdout


@pytest.mark.parametrize(
    "replacement",
    [
        "self.n = n\n        self.n += 1",
        "if n > 0:\n            self.n = n",
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_unproved_field_extent_aliases(tmp_path, monkeypatch, replacement):
    package = _source_package(tmp_path, monkeypatch, FIELD_SOURCE.replace("self.n = n", replacement))
    with pytest.raises(ti.aot.PackageExportError, match="Field extents require"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_keeps_actual_host_field_extent_expressions(tmp_path, monkeypatch):
    source = FIELD_SOURCE.replace("self.n = n", "self.n = n\n        self.bump()")
    source += "\n    def bump(self):\n        self.n += 1\n"
    package = _source_package(tmp_path, monkeypatch, source)
    result = ti.aot.export_package(package, tmp_path / "sdk")
    original = package.Model(7, 3)
    original.step(1.0)
    with Runtime(result["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as library:
        model = library.host.Model(7, 3)
        assert model.a.shape == (8, 3) and model.v.shape == (7,)
        model.step(1.0)
        np.testing.assert_array_equal(np.frombuffer(model.a.read(), np.float32).reshape(8, 3), original.a.to_numpy())


@pytest.mark.parametrize(
    "allocation,diagnostic",
    [
        ("ti.field(ti.f32, shape=2147483648)", "positive i32 dimension"),
        ("ti.field(ti.types.struct(x=ti.f32), shape=())", "scalar/vector/matrix elements"),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_field_preflight_rejection_preserves_normal_jit(tmp_path, monkeypatch, allocation, diagnostic):
    source = FIELD_SOURCE.replace("ti.field(ti.f32, shape=())", allocation)
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ValueError, match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()
    # Earlier fields in the rejected owner must not leave a pending builder or
    # unplaced fields that break later, otherwise unrelated JIT execution.
    output = ti.field(ti.i32, shape=3)

    @ti.kernel
    def fill():
        for i in output:
            output[i] = i + 7

    fill()
    np.testing.assert_array_equal(output.to_numpy(), [7, 8, 9])


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_absolute_and_lazy_imports_keep_their_code_owner(tmp_path, monkeypatch):
    source = SOURCE.replace("from .device_helpers import scale", "from definition_fixture.device_helpers import scale")
    source += """

def imports(x):
    import definition_fixture.unimported
    import definition_fixture.unimported as alias
    from definition_fixture import unimported as from_root
    from definition_fixture.unimported import available_without_a_test as call
    return (definition_fixture.unimported.available_without_a_test(x)
            + alias.available_without_a_test(x) + from_root.available_without_a_test(x) + call(x))
"""
    package = _source_package(tmp_path, monkeypatch, source)
    ti.aot.export_package(package, tmp_path / "sdk")
    probe = tmp_path / "consume_imports.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        class NoFrontend(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError('Private frontend/source import: ' + fullname)
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime('cpu') as runtime, load_package(runtime, sys.argv[1]) as a:
            with load_package(runtime, sys.argv[1]) as b:
                assert a.host.model.imports(2) == 40
                assert b.host.model.imports(3) == 60
                assert a.host.unimported is not b.host.unimported
                assert a.host.unimported.available_without_a_test.package is a
                assert b.host.unimported.available_without_a_test.package is b
                assert 'definition_fixture' not in sys.modules
            assert a.host.model.imports(4) == 80
        print('PACKAGE_IMPORT_ISOLATION_OK')
        """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PACKAGE_IMPORT_ISOLATION_OK" in result.stdout


HOST_BUFFER_SOURCE = """
import taichi_forge as ti

@ti.kernel
def update(source: ti.types.ndarray(ti.f32, 1), output: ti.types.ndarray(ti.f32, 1), offset: ti.f32):
    for i in source:
        output[i] = source[i] + offset
        source[i] += 1

@ti.kernel
def transform(values: ti.types.ndarray(ti.types.vector(3, ti.f32), 1)):
    for i in values:
        values[i] += ti.Vector([1, 2, 3])
"""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_host_buffers_preserve_types_and_reject_unproved_aliases(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, HOST_BUFFER_SOURCE, initializer="from . import model\n")
    original = sys.modules["definition_fixture.model"]
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as library:
        resources = set(runtime._resources)
        for count in (7, 19):
            expected = np.arange(count, dtype=np.float32)
            expected_output = np.zeros(count, dtype=np.float32)
            original.update(expected, expected_output, 2.0)
            actual = np.arange(count, dtype=np.float32)
            output = np.zeros(count, dtype=np.float32)
            library.host.model.update(actual, output, 2.0)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(output, expected_output)
            separate = np.zeros(count, dtype=np.float32)
            library.host.model.update(actual, separate, -4.0)
            np.testing.assert_array_equal(separate, expected - 4)
            np.testing.assert_array_equal(actual, expected + 1)
            vectors = np.zeros((count, 3), dtype=np.float32)
            library.host.model.transform(vectors)
            np.testing.assert_array_equal(vectors, [[1, 2, 3]] * count)
            assert runtime._resources == resources

        original_data = np.arange(9, dtype=np.float32)
        for a, b in ((original_data, original_data.view()), (original_data[:-1], original_data[1:])):
            with pytest.raises(ValueError, match="Aliased or overlapping Host arrays"):
                library.host.model.update(a, b, 3.0)
        np.testing.assert_array_equal(original_data, np.arange(9, dtype=np.float32))
        with pytest.raises(ValueError, match="dtype"):
            library.host.model.update(np.zeros(7, dtype=np.float64), np.zeros(7, dtype=np.float32), 1.0)
        with pytest.raises(ValueError, match="C-contiguous"):
            library.host.model.update(original_data[::2], original_data[::2], 1.0)
        with pytest.raises(ValueError, match="element shape"):
            library.host.model.transform(np.zeros((7, 2), dtype=np.float32))
        assert runtime._resources == resources


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_collects_local_dependencies_without_tracing_and_isolates_dynamic_imports(tmp_path, monkeypatch):
    source_root = tmp_path / "src"
    source_root.mkdir()
    dependencies = {
        "sibling_fixture/__init__.py": "initialized = 31\n",
        "sibling_fixture/ops.py": """import taichi_forge as ti
@ti.kernel
def extra(x: ti.i32) -> ti.i32:
    return x * 11
@ti.kernel
def never_called(x: ti.i32) -> ti.i32:
    return x * 13
""",
        "sibling_fixture/lazy.py": """import taichi_forge as ti
@ti.kernel
def calc(x: ti.i32) -> ti.i32:
    return x * 17
""",
        "sibling_fixture/not_selected.py": "raise AssertionError('must not import unspecified dynamic target')\n",
        "ns_fixture/helper.py": "def label(value):\n    return 'item-' + str(value)\n",
    }
    for name, text in dependencies.items():
        path = source_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf8")
    source = SOURCE + """

def use_dependencies(x):
    import sibling_fixture.ops
    from ns_fixture import helper
    return sibling_fixture.ops.extra(x), sibling_fixture.initialized, helper.label(x)

from importlib import import_module as late
import importlib as imports
import importlib.util
def literal_late(x):
    return late('sibling_fixture.lazy').calc(x)
def dynamic(name):
    return imports.import_module(name)
def relative_late(x):
    return importlib.import_module('.lazy', 'sibling_fixture').calc(x)
"""
    package = _source_package(source_root, monkeypatch, source)
    with pytest.raises(ValueError, match="outside its application source root"):
        ti.aot.export_package(package, source_root / "sdk", source_root=source_root)
    manifest = ti.aot.export_package(package, tmp_path / "sdk", source_root=source_root)
    assert manifest["version"] == 2
    assert "sibling_fixture.not_selected" not in manifest["application_modules"]
    assert "ns_fixture" in manifest["application_modules"]
    assert {"sibling_fixture.ops.never_called", "sibling_fixture.lazy.calc"} <= {
        entry["definition"] for entry in manifest["entries"].values()
    }
    assert any(edge["kind"] == "literal_dynamic" for edge in manifest["dependencies"]["import_edges"])
    assert manifest["dependencies"]["dynamic_import_sites"]
    probe = tmp_path / "consume_dependencies.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        class NoPrivateSource(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture', 'sibling_fixture', 'ns_fixture'):
                    raise AssertionError('Private source import: ' + fullname)
        sys.meta_path.insert(0, NoPrivateSource())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime('cpu') as runtime, load_package(runtime, sys.argv[1]) as a:
            model = a.host.model
            assert model.use_dependencies(3) == (33, 31, 'item-3')
            assert model.literal_late(2) == 34
            assert model.relative_late(3) == 51
            with load_package(runtime, sys.argv[1]) as b:
                left = model.dynamic('sibling_fixture.ops')
                right = b.host.model.dynamic('sibling_fixture.ops')
                assert left is not right and left.extra.package is a and right.extra.package is b
                assert model.dynamic('math').sqrt(9) == 3
            assert left.never_called(4) == 52
            try:
                model.dynamic('sibling_fixture.not_selected')
            except ImportError as error:
                assert 'was not exported' in str(error)
            else:
                raise AssertionError('Unresolved local dependency fell back to source')
        print('LOCAL_DEPENDENCIES_SOURCE_FREE_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LOCAL_DEPENDENCIES_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize("storage", ["field", "ndarray"])
@pytest.mark.parametrize("helper_kind", ["func", "real_func", "pyfunc"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_inheritance_uses_actual_method_dependencies_and_shared_code(
    tmp_path, monkeypatch, storage, helper_kind
):
    if helper_kind == "real_func" and ti.cfg.arch == ti.vulkan:
        pytest.skip("Vulkan does not support real functions")
    source = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, gain: float):
        global constructors
        constructors += 1
        self.n = n
        self.gain = gain
        self.values = ti.field(ti.f32, shape=n)
        self.step(1.0)

    @ti.func
    def affine(self, x: ti.f32) -> ti.f32:
        return x * self.gain

    @ti.kernel
    def step(self, amount: ti.f32):
        for i in self.values:
            self.values[i] = self.affine(i) + amount

    @ti.kernel
    def size(self) -> ti.i32:
        return self.n
"""
    if storage == "ndarray":
        source = source.replace("ti.field(ti.f32, shape=n)", "ti.ndarray(ti.f32, shape=n)")
        source = source.replace("self.step(1.0)", "self.step(self.values, 1.0)")
        source = source.replace(
            "def step(self, amount:", "def step(self, values: ti.types.ndarray(dtype=ti.f32, ndim=1), amount:"
        )
        source = source.replace("for i in self.values:", "for i in values:").replace("self.values[i] =", "values[i] =")
    methods = source[source.index("    @ti.kernel\n    def step") :]
    source = source[: source.index("    @ti.kernel\n    def step")]
    constructor = source[source.index("    def __init__") : source.index("    @ti.func")]
    source = source.replace(constructor, "")
    source = source.replace(
        "@ti.data_oriented\nclass Model:",
        "class Storage:\n"
        + constructor
        + "\nclass KernelMethods:\n"
        + methods
        + "\n@ti.data_oriented\nclass Model(Storage, KernelMethods):",
    )
    source = source.replace("@ti.func", "@ti." + helper_kind)
    package = _source_package(tmp_path, monkeypatch, source)
    derived = tmp_path / "definition_fixture/derived.py"
    derived.write_text(
        textwrap.dedent("""
        import taichi_forge as ti
        from .model import Model

        class Unchanged(Model):
            pass

        class Offset:
            @ti.func
            def affine(self, x: ti.f32) -> ti.f32:
                return x * self.gain + 10.0

        class Changed(Offset, Model):
            def __init__(self, n: int, gain: float):
                super().__init__(n, gain)
                self.host_only = 23

        class Fails(Model):
            def __init__(self, n: int, gain: float):
                super().__init__(n, gain)
                raise ValueError('after base allocation')
        """).replace("@ti.func", "@ti." + helper_kind),
        encoding="utf8",
    )
    monkeypatch.delitem(sys.modules, "definition_fixture.derived", raising=False)
    assert "definition_fixture.derived" not in sys.modules
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    base = sys.modules["definition_fixture.model"]
    assert manifest["version"] == 3
    classes = sys.modules["definition_fixture.derived"]
    assert base.constructors == 0
    # A new class/Host field does not create another compiled numerical unit.
    by_name = {name: entry for entry in manifest["entries"].values() for name in entry["definitions"]}
    original = by_name["definition_fixture.model.Model.step"]
    assert by_name["definition_fixture.derived.Unchanged.step"] is original
    assert by_name["definition_fixture.derived.Fails.step"] is original
    assert by_name["definition_fixture.derived.Changed.step"] is not original
    assert by_name["definition_fixture.derived.Changed.size"] is by_name["definition_fixture.model.Model.size"]
    if ti.cfg.arch == ti.cpu and helper_kind == "func" and storage == "field":
        base_manifest = ti.aot.export_package(base, tmp_path / "base_sdk")
        base_symbols = {item["definition"]: item["symbol"] for item in base_manifest["entries"].values()}
        assert base_symbols["definition_fixture.model.Model.step"] == original["symbol"]
        assert base.constructors == 0
    with zipfile.ZipFile(tmp_path / "sdk/code.tcm") as archive:
        code = [item for item in archive.infolist() if item.filename.endswith((".ll", ".bc", ".spv"))]
        code_count = len(code)
        assert sum(item.file_size for item in code) > 0
    expected = classes.Changed(7, 2.0).values.to_numpy()
    np.testing.assert_array_equal(expected, np.arange(7, dtype=np.float32) * 2 + 11)
    probe = tmp_path / "consume_inheritance.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib
        import importlib.abc
        import sys
        import numpy as np
        class NoPrivateSource(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoPrivateSource())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            derived = importlib.import_module(code.host.__name__ + '.derived')
            a, b = derived.Changed(7, 2.0), derived.Unchanged(19, -3.0)
            read = lambda owner: np.frombuffer(owner.values.read(), np.float32)
            np.testing.assert_array_equal(read(a), np.arange(7, dtype=np.float32) * 2 + 11)
            np.testing.assert_array_equal(read(b), np.arange(19, dtype=np.float32) * -3 + 1)
            assert a.size() == 7 and b.size() == 19 and a.host_only == 23
            if sys.argv[3] == 'field':
                assert a.values._group_owner is not b.values._group_owner
                a.step(4.0)
            else:
                a.step(a.values, 4.0)
            np.testing.assert_array_equal(read(a), np.arange(7, dtype=np.float32) * 2 + 14)
            np.testing.assert_array_equal(read(b), np.arange(19, dtype=np.float32) * -3 + 1)
            registered = set(runtime._resources)
            try:
                derived.Fails(3, 2.0)
            except ValueError as error:
                assert str(error) == 'after base allocation'
            else:
                raise AssertionError('Original constructor failure disappeared')
            assert runtime._resources == registered
        print('INHERITANCE_SOURCE_FREE_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"], storage],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "INHERITANCE_SOURCE_FREE_OK" in result.stdout
    assert code_count == (6 if helper_kind == "pyfunc" else 4)  # changed helper affects only step and public pyfunc


@pytest.mark.parametrize(
    "constructor,diagnostic",
    [
        ("if n > 0:\n            super().__init__(n, m)", "one unconditional"),
        ("super().__init__(n, m)\n        self.other = ti.field(ti.f32, shape=n)", "multiple constructors"),
        ("super().__init__(n, m)\n        self.a = ti.ndarray(ti.f32, shape=n)", "inherited resource/type replacement"),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_unproven_inherited_resource_construction(tmp_path, monkeypatch, constructor, diagnostic):
    source = (
        FIELD_SOURCE + "\nclass Derived(Model):\n    def __init__(self, n: int, m: int):\n        " + constructor + "\n"
    )
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()
    assert sys.modules["definition_fixture.model"].constructors == 0


@pytest.mark.parametrize(
    "host,diagnostic",
    [
        ("super().step(1.0)", "explicit super numerical calls"),
        ("Model.step(self, 1.0)", "unbound numerical references"),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_unbound_inherited_numerical_calls(tmp_path, monkeypatch, host, diagnostic):
    source = FIELD_SOURCE + "\nclass Derived(Model):\n    def drive(self):\n        " + host + "\n"
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_numerical_mixin_without_concrete_receiver(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, FIELD_SOURCE.replace("@ti.data_oriented\n", ""))
    with pytest.raises(ti.aot.PackageExportError, match="no exported concrete"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_preserves_nonreturning_interfaces_without_numerical_payload(tmp_path, monkeypatch):
    source = """
from __future__ import annotations
import taichi_forge as ti
from typing import NoReturn as NeverReturns
calls = 0

def unsupported(message: str) -> NeverReturns:
    global calls
    calls += 1
    raise NotImplementedError(message)

def broken_contract() -> 'NeverReturns':
    return None

@ti.data_oriented
class Model:
    @ti.kernel
    def clear(self):
        unsupported('implement clear')

    @ti.func
    def missing(self, x):
        unsupported('implement missing')

    @ti.pyfunc
    def alternative(self, x):
        unsupported('implement alternative')

    def drive(self):
        self.missing(3)

class Inherited(Model):
    pass

class Concrete(Model):
    @ti.kernel
    def clear(self) -> ti.i32:
        return 17

@ti.func
def unavailable():
    unsupported('unavailable')

@ti.kernel
def invalid_contract():
    broken_contract()
"""
    # Forge annotations on typed numerical entries are evaluated normally;
    # postpone only the ordinary Host helpers' NoReturn annotations.
    source = source.replace("from __future__ import annotations\n", "").replace(
        "-> NeverReturns:", "-> 'NeverReturns':"
    )
    package = _source_package(tmp_path, monkeypatch, source)
    (tmp_path / "definition_fixture/caller.py").write_text(
        "from .model import unavailable\ndef call():\n    unavailable()\n", encoding="utf8"
    )
    monkeypatch.delitem(sys.modules, "definition_fixture.caller", raising=False)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    original = sys.modules["definition_fixture.model"]
    assert original.calls == 0
    assert original.Concrete().clear() == 17
    assert len(manifest["entries"]) == 2  # concrete implementation and unimported typed kernel
    assert "definition_fixture.model.Model.clear" in manifest["error_interfaces"]
    assert "definition_fixture.model.Inherited.clear" in manifest["error_interfaces"]
    host = (tmp_path / "sdk/host/model.py").read_text(encoding="utf8")
    assert "implement clear" in host and "return 17" not in host
    probe = tmp_path / "consume_errors.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib
        import importlib.abc
        import sys
        class NoPrivateSource(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoPrivateSource())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime('cpu') as runtime, load_package(runtime, sys.argv[1]) as code:
            model = importlib.import_module(code.host.__name__ + '.model')
            caller = importlib.import_module(code.host.__name__ + '.caller')
            for operation, message in [
                (model.Model().clear, 'implement clear'),
                (model.Inherited().clear, 'implement clear'),
                (model.Inherited().drive, 'implement missing'),
                (lambda: model.Concrete().alternative(1), 'implement alternative'),
                (caller.call, 'unavailable'),
            ]:
                try:
                    operation()
                except NotImplementedError as error:
                    assert str(error) == message
                else:
                    raise AssertionError('Error interface silently succeeded')
            assert model.calls == 5 and model.Concrete().clear() == 17
            try:
                model.invalid_contract()
            except RuntimeError as error:
                assert 'returned unexpectedly' in str(error)
            else:
                raise AssertionError('NoReturn contract silently returned')
        print('ERROR_INTERFACES_SOURCE_FREE_OK')
        """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ERROR_INTERFACES_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize(
    "body,annotation",
    [
        ("unsupported(1 + 2)", "NoReturn"),
        ("value = 3\n        unsupported('missing')", "NoReturn"),
        ("unsupported('missing')", "None"),
        ("unsupported('missing')", "invented_NoReturn"),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_does_not_misclassify_computation_or_unproven_error_names(tmp_path, monkeypatch, body, annotation):
    source = f"""
import taichi_forge as ti
from typing import NoReturn
invented_NoReturn = object()
def unsupported(value) -> '{annotation}':
    raise NotImplementedError(value)
@ti.data_oriented
class Model:
    @ti.kernel
    def clear(self):
        {body}
"""
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="undeclared global/closure capture 'unsupported'"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()


SCOPE_VALUES_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, gain, shift, transform):
        global constructors
        constructors += 1
        self.gain, self.shift, self.transform = gain, shift, transform
        self.unused = 11
        scope = ti.experimental.ResourceScope(self)
        scope.value('gain', ti.f32)
        scope.value('shift', ti.types.vector(2, ti.f32))
        scope.value('transform', ti.types.matrix(2, 2, ti.f32))
        scope.value('unused', ti.i32)

    @ti.kernel
    def compute(self, x: ti.f32) -> ti.f32:
        value = self.transform @ ti.Vector([x, self.gain]) + self.shift
        return value[0] + value[1]

    @ti.kernel
    def independent(self, x: ti.i32) -> ti.i32:
        return x + 5
"""


@pytest.mark.parametrize("host_tensor", [False, True])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_resource_scope_values_preserve_compound_captures(tmp_path, monkeypatch, host_tensor):
    source = SCOPE_VALUES_SOURCE
    if host_tensor:
        source = "from taichi_forge import Matrix as MatrixValue\n" + source.replace(
            "self.gain, self.shift, self.transform = gain, shift, transform",
            "self.gain = gain\n        self.shift = ti.Vector(shift, dt=ti.f32)\n        self.transform = MatrixValue(transform, ti.f32)",
        )
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    original = sys.modules["definition_fixture.model"]
    assert manifest["version"] == (8 if host_tensor else 5) and original.constructors == 0
    entries = {entry["definition"].rsplit(".", 1)[-1]: entry for entry in manifest["entries"].values()}
    captures = {item["name"]: item for item in entries["compute"]["captures"]}
    assert set(captures) == {"gain", "shift", "transform"}
    assert captures["shift"]["element_shape"] == [2]
    assert captures["transform"]["element_shape"] == [2, 2]
    assert not entries["independent"]["captures"]
    shift, transform = [1.0, 3.0], [[2.0, 1.0], [0.0, 4.0]]
    reference = original.Model(
        2.0, shift if host_tensor else ti.Vector(shift), transform if host_tensor else ti.Matrix(transform)
    )
    assert reference.compute(3.0) == 20.0
    probe = tmp_path / "consume_scope_values.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        class NoPrivateSource(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoPrivateSource())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a = code.host.Model(2.0, [1.0, 3.0], [[2.0, 1.0], [0.0, 4.0]])
            b = code.host.Model(-1.0, [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
            assert a.compute(3.0) == 20.0 and b.compute(3.0) == 2.0
            if sys.argv[3] == 'True':
                assert a.shift.get_shape() == (2,) and a.transform.get_shape() == (2, 2)
                assert a.shift.x == 1 and a.shift.y == 3
                assert a.transform[0, 1] == 1
                assert a.shift.to_numpy().dtype.name == 'float32'
                assert a.transform.to_list() == [[2, 1], [0, 4]]
                a.shift.x = 2
                assert a.compute(3.0) == 21.0 and b.compute(3.0) == 2.0
                try:
                    a.shift + 1
                except TypeError:
                    pass
                else:
                    raise AssertionError('Unimplemented Host algebra must not silently change semantics')
            a.shift = [10.0, -3.0]
            assert a.compute(3.0) == 23.0 and b.compute(3.0) == 2.0
            assert a.independent(7) == 12
        print('SCOPE_VALUES_SOURCE_FREE_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"], str(host_tensor)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SCOPE_VALUES_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize(
    "replacement,diagnostic",
    [
        ("if gain > 0:\n            scope.value('gain', ti.f32)", "unconditional supported resource declaration"),
        ("alias = scope\n        scope.value('gain', ti.f32)", "unconditional supported resource declaration"),
        ("scope = ti.experimental.ResourceScope(self)\n        scope.value('gain', ti.f32)", "must not be reassigned"),
        ("scope.array = scope.field(ti.f32, shape=2, placement=None)", "placement must retain"),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_resource_scope_rejects_unproven_roles(tmp_path, monkeypatch, replacement, diagnostic):
    package = _source_package(
        tmp_path, monkeypatch, SCOPE_VALUES_SOURCE.replace("scope.value('gain', ti.f32)", replacement)
    )
    with pytest.raises(ti.aot.PackageExportError, match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


SCOPE_FIELDS_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, m: int):
        global constructors
        constructors += 1
        self.n, self.m = n, m
        unrelated_host_value = 2**48
        unrelated_host_value = 0
        scope = ti.experimental.ResourceScope(self)
        rows, cols = scope.dim('n'), scope.dim('m')
        count = (rows + 1) * 2
        scope.a = scope.field(ti.f32, shape=count, initialization='zero')
        self.total = ti.field(ti.f32, shape=())
        scope.v = scope.vector_field(2, ti.f32, shape=(rows.maximum(1), cols))
        scope.mat = scope.matrix_field(2, 2, ti.f32, shape=())

    @ti.kernel
    def step(self, gain: ti.f32):
        for i in self.a:
            self.a[i] = gain * (i + self.n)
        for i, j in self.v:
            self.v[i, j] = ti.Vector([gain * i, gain * j])
        self.mat[None] = ti.Matrix([[gain, 2.0], [3.0, gain + 4.0]])
        self.total[None] = gain * (self.n + self.m)
"""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_scope_fields_share_dimension_and_native_layout_rules(tmp_path, monkeypatch):
    from tests.python.test_aot_dense_snode_group import _compiled_nodes

    package = _source_package(tmp_path, monkeypatch, SCOPE_FIELDS_SOURCE)
    original = package.Model(7, 3)
    original.step(2.0)
    expected = {name: getattr(original, name).to_numpy() for name in ("a", "v", "mat", "total")}
    root = ti.lang.impl.get_runtime().prog.get_snode_root(original.a.snode.ptr.get_snode_tree_id())
    layout = _compiled_nodes(root)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 5
    assert sys.modules["definition_fixture.model"].constructors == 1
    assert len(manifest["field_groups"]) == 1
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as library:
        a, b = library.host.Model(7, 3), library.host.Model(0, 5)
        assert a.a._group_owner.layout["nodes"] == layout
        assert a.a._group_owner is a.v._group_owner is a.mat._group_owner is a.total._group_owner
        assert a.a._group_owner is not b.a._group_owner
        assert a.a.shape == (16,) and b.a.shape == (2,) and b.v.shape == (1, 5)
        a.step(2.0)
        b.step(-3.0)
        for name, values in expected.items():
            np.testing.assert_array_equal(
                np.frombuffer(getattr(a, name).read(), np.float32).reshape(values.shape), values
            )
        np.testing.assert_array_equal(np.frombuffer(b.a.read(), np.float32), [0.0, -3.0])
        assert np.frombuffer(b.total.read(), np.float32).item() == -15.0
        registered = set(runtime._resources)
        with pytest.raises((ValueError, RuntimeError)):
            library.host.Model(-1, 2)
        assert runtime._resources == registered
    probe = tmp_path / "consume_scope_fields.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class NoPrivateSource(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoPrivateSource())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a, b = code.host.Model(3, 2), code.host.Model(19, 1)
            a.step(2.0)
            b.step(-1.0)
            np.testing.assert_array_equal(np.frombuffer(a.a.read(), np.float32), 2 * (np.arange(8) + 3))
            np.testing.assert_array_equal(np.frombuffer(b.a.read(), np.float32), -(np.arange(40) + 19))
            assert a.v.shape == (3, 2) and b.v.shape == (19, 1)
            assert a.a._group_owner is not b.a._group_owner
        print('SCOPE_FIELDS_SOURCE_FREE_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SCOPE_FIELDS_SOURCE_FREE_OK" in result.stdout


@pytest.mark.parametrize(
    "before,after,diagnostic",
    [
        ("count = (rows + 1) * 2", "count = n * 2", "shape requires declared dimensions"),
        (
            "count = (rows + 1) * 2",
            "count = (rows + 1) * 2\n        count = rows",
            "shape requires declared dimensions",
        ),
        (
            "self.total = ti.field(ti.f32, shape=())",
            "self.a = 7\n        self.total = ti.field(ti.f32, shape=())",
            "one unconditional owner attribute",
        ),
        (
            "rows, cols = scope.dim('n'), scope.dim('m')",
            "if n > 0:\n            rows, cols = scope.dim('n'), scope.dim('m')",
            "shape requires declared dimensions",
        ),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_scope_fields_reject_unproven_dimensions_and_replacements(
    tmp_path, monkeypatch, before, after, diagnostic
):
    package = _source_package(tmp_path, monkeypatch, SCOPE_FIELDS_SOURCE.replace(before, after))
    with pytest.raises(ti.aot.PackageExportError, match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()
    assert sys.modules["definition_fixture.model"].constructors == 0


EXPLICIT_GROUP_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, m: int):
        global constructors
        constructors += 1
        self.n, self.m = n, m
        fields = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=fields, initialization='zero')
        other = scope.dense(ti.i, scope.dim('m'))
        cells = scope.dense(ti.i, scope.dim('n'))
        scope.a = scope.field(ti.f32, shape=scope.dim('n'), placement=cells)
        scope.v = scope.vector_field(2, ti.f64, shape=scope.dim('n'), placement=cells)
        scope.extra = scope.field(ti.i32, shape=scope.dim('m'), placement=other)
        self.tree = fields.finalize()
        separate = ti.FieldsBuilder()
        scalar = ti.experimental.ResourceScope(self, builder=separate)
        scalar.total = scalar.field(ti.f32, shape=())
        scalar.independent = scalar.field(ti.f32, shape=scalar.dim('n'))
        self.scalar_tree = separate.finalize()
        self.ordinary = ti.field(ti.i32, shape=1)

    @ti.kernel
    def step(self, gain: ti.f32):
        for i in self.a:
            self.a[i] = gain * i
            self.v[i] = ti.Vector([i, gain])
        for i in self.extra:
            self.extra[i] = i + 9
        self.total[None] = gain
        self.ordinary[0] = 7

    @ti.kernel
    def summary(self, gain: ti.f32):
        self.total[None] += gain

    def destroy_fields(self):
        self.tree.destroy()
"""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_explicit_groups_preserve_placement_and_lifetime(tmp_path, monkeypatch):
    from tests.python.test_aot_dense_snode_group import _compiled_nodes

    # The current Vulkan AOT target does not enable f64; retain f64 coverage
    # on LLVM and exercise the same placement/ownership rules with f32 there.
    source = EXPLICIT_GROUP_SOURCE.replace("ti.f64", "ti.f32") if ti.cfg.arch == ti.vulkan else EXPLICIT_GROUP_SOURCE
    package = _source_package(tmp_path, monkeypatch, source)
    original = package.Model(7, 3)
    original.step(2.0)
    expected = {name: getattr(original, name).to_numpy() for name in ("a", "v", "extra", "total", "ordinary")}
    program = ti.lang.impl.get_runtime().prog
    layouts = {
        name: _compiled_nodes(program.get_snode_root(getattr(original, name).snode.ptr.get_snode_tree_id()))
        for name in ("a", "total", "ordinary")
    }
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert manifest["version"] == 10
    assert len(manifest["field_groups"]) == 3
    assert sys.modules["definition_fixture.model"].constructors == 1
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        a, b = code.host.Model(7, 3), code.host.Model(19, 5)
        for name, layout in layouts.items():
            assert getattr(a, name)._group_owner.layout["nodes"] == layout
        assert a.a._group_owner is a.v._group_owner is a.extra._group_owner
        assert a.a._group_owner is not a.total._group_owner
        a.step(2.0)
        b.step(-1.0)
        for name, data in expected.items():
            np.testing.assert_array_equal(getattr(a, name).to_numpy(), data)
        a.destroy_fields()
        a.summary(3.0)
        assert a.total.to_numpy().item() == 5.0
        with pytest.raises(RuntimeError):
            a.a.to_numpy()
        b.step(-2.0)
        np.testing.assert_array_equal(b.a.to_numpy(), -2 * np.arange(19))
        registered = set(runtime._resources)
        with pytest.raises((ValueError, RuntimeError)):
            code.host.Model(0, 3)
        assert set(runtime._resources) == registered
    probe = tmp_path / "consume_explicit_groups.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, Block())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a, b = code.host.Model(3, 2), code.host.Model(19, 4)
            a.step(2.)
            b.step(-1.)
            np.testing.assert_array_equal(a.a.to_numpy(), np.arange(3)*2)
            a.destroy_fields()
            a.summary(7.)
            assert a.total.to_numpy().item() == 9.
            np.testing.assert_array_equal(b.a.to_numpy(), -np.arange(19))
        print('EXPLICIT_GROUPS_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EXPLICIT_GROUPS_OK" in result.stdout


@pytest.mark.parametrize(
    "before,after,diagnostic",
    [
        ("placement=cells)", "placement=other)", "group and extent relations"),
        (
            "self.tree = fields.finalize()",
            "leak = fields\n        self.tree = fields.finalize()",
            "escaping/reassigned",
        ),
        ("self.tree = fields.finalize()", "self.tree = fields.finalize()\n        fields.finalize()", "finalize once"),
        (
            "cells = scope.dense(ti.i, scope.dim('n'))",
            "cells = scope.dense(ti.j, scope.dim('n'))",
            "canonical ordered axes",
        ),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_explicit_groups_reject_ambiguous_ownership(tmp_path, monkeypatch, before, after, diagnostic):
    package = _source_package(tmp_path, monkeypatch, EXPLICIT_GROUP_SOURCE.replace(before, after))
    with pytest.raises(ti.aot.PackageExportError, match=diagnostic):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()
    assert sys.modules["definition_fixture.model"].constructors == 0


@pytest.mark.parametrize(
    "before,after,diagnostic",
    [
        ("scope.a =", "self.n += 1\n        scope.a =", "placement changed"),
        ("scalar = ti.experimental", "self.n += 1\n        scalar = ti.experimental", "Shared dimension changed"),
    ],
)
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_explicit_placement_keeps_its_construction_snapshot(tmp_path, monkeypatch, before, after, diagnostic):
    source = EXPLICIT_GROUP_SOURCE.replace(before, after)
    package = _source_package(tmp_path, monkeypatch, source)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as code:
        registered = set(runtime._resources)
        with pytest.raises(ValueError, match=diagnostic):
            code.host.Model(7, 3)
        assert set(runtime._resources) == registered


RESOURCE_FACTORY_SOURCE = """
import taichi_forge as ti
effects = []
def resource_scope(owner, *, builder=None, requires=(), initialization='consumer_explicit'):
    if initialization not in ('consumer_explicit', 'zero'):
        raise ValueError('Unknown resource initialization')
    factory = getattr(getattr(ti, 'experimental', None), 'ResourceScope', None)
    qualified = factory is not None and all(hasattr(factory, name) for name in requires)
    if qualified and initialization != 'consumer_explicit':
        import inspect
        qualified = 'initialization' in inspect.signature(factory).parameters
        if qualified:
            return factory(owner, builder=builder, initialization=initialization)
    return factory(owner, builder=builder) if qualified else object()
"""


def _resource_factory_fixture(
    tmp_path,
    monkeypatch,
    factory_source=RESOURCE_FACTORY_SOURCE,
    *,
    requires="dense",
    extra_argument="",
    model_source=EXPLICIT_GROUP_SOURCE,
):
    (tmp_path / "resource_factories.py").write_text(textwrap.dedent(factory_source), encoding="utf8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "resource_factories", raising=False)
    source = model_source.replace(
        "import taichi_forge as ti", "import taichi_forge as ti\nfrom resource_factories import resource_scope"
    )
    source = source.replace(
        "ti.experimental.ResourceScope(self, builder=fields, initialization='zero')",
        f"resource_scope(self, {extra_argument}builder=fields, requires=('{requires}',), initialization='zero')",
    )
    source = source.replace(
        "ti.experimental.ResourceScope(self, builder=separate)", "resource_scope(self, builder=separate)"
    )
    if ti.cfg.arch == ti.vulkan:
        source = source.replace("ti.f64", "ti.f32")
    return _source_package(tmp_path, monkeypatch, source)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_proves_imported_resource_factory_without_running_it(tmp_path, monkeypatch):
    package = _resource_factory_fixture(tmp_path, monkeypatch)
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert len(manifest["declaration_factories"]) == 2
    assert {d["definition"] for d in manifest["declaration_factories"]} == {"resource_factories.resource_scope"}
    host = (tmp_path / "sdk/host/model.py").read_text()
    assert "from resource_factories import" not in host
    original = package.Model(7, 3)
    original.step(2.0)
    probe = tmp_path / "consume_factory.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi','taichi_forge','definition_fixture','resource_factories'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, Block())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a,b = code.host.Model(7,3),code.host.Model(19,2)
            a.step(2.)
            b.step(-1.)
            np.testing.assert_array_equal(a.a.to_numpy(), 2*np.arange(7))
            np.testing.assert_array_equal(b.a.to_numpy(), -np.arange(19))
        print('RESOURCE_FACTORY_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESOURCE_FACTORY_OK" in result.stdout


@pytest.mark.parametrize("change", ["effect", "model", "global", "missing", "argument", "dropped", "early_return"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_unproven_resource_factory_paths(tmp_path, monkeypatch, change):
    source = RESOURCE_FACTORY_SOURCE
    if change == "effect":
        source = source.replace("    if initialization", "    effects.append(1)\n    if initialization", 1)
    if change == "model":
        source = source.replace("    factory =", "    if owner.n > 3:\n        effects.append(1)\n    factory =", 1)
    if change == "global":
        source = "runtime_choice = True\n" + source.replace(
            "    factory =", "    if runtime_choice:\n        effects.append(1)\n    factory =", 1
        )
    if change == "early_return":
        source = source.replace(
            "    factory =", "    if initialization == 'zero':\n        return None\n    factory =", 1
        )
    extra_argument = ""
    if change in ("argument", "dropped"):
        source = source.replace("requires=(),", "requires=(), ignored=None,")
        extra_argument = "ignored=self.observe(), " if change == "argument" else "ignored=self, "
    package = _resource_factory_fixture(
        tmp_path,
        monkeypatch,
        source,
        requires="missing" if change == "missing" else "dense",
        extra_argument=extra_argument,
    )
    with pytest.raises(ti.aot.PackageExportError, match="factory forwarding is unproven"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["resource_factories"].effects == []
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@pytest.mark.parametrize("binding", ["parameter", "assignment"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_shadowed_resource_factory(tmp_path, monkeypatch, binding):
    source = EXPLICIT_GROUP_SOURCE
    if binding == "parameter":
        source = source.replace("n: int, m: int):", "n: int, m: int, resource_scope=None):")
    else:
        source = source.replace(
            "fields = ti.FieldsBuilder()", "resource_scope = None\n        fields = ti.FieldsBuilder()"
        )
    package = _resource_factory_fixture(tmp_path, monkeypatch, model_source=source)
    with pytest.raises(ti.aot.PackageExportError, match="factory binding is shadowed"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


STATIC_DOMAIN_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, first: bool, second: bool, third: bool, unused: int):
        global constructors
        constructors += 1
        self.n = n
        self.first, self.second, self.third, self.unused = first, second, third, unused
        scope = ti.experimental.ResourceScope(self)
        scope.static('first', choices=(False, True))
        scope.static('second', choices=(False, True), kernel_only=True)
        scope.static('third', choices=(False, True), kernel_only=True)
        scope.static('unused', choices=(0, 1, 2, 3, 4, 5, 6, 7))
        self.values = ti.field(ti.i32, shape=n)

    @ti.func
    def mode(self):
        result = 0
        if ti.static(self.first):
            if ti.static(self.second):
                result = 10
            else:
                result = 20
        else:
            if ti.static(self.third):
                result = 30
            else:
                result = 40
        return result

    @ti.kernel
    def compute(self):
        for i in range(self.n):
            self.values[i] = self.mode() + i

    @ti.kernel
    def other(self):
        for i in range(self.n):
            if ti.static(self.second):
                self.values[i] = 5
            else:
                self.values[i] = -5

    @ti.kernel
    def plain(self):
        for i in range(self.n):
            self.values[i] = i

class Equivalent(Model):
    pass
"""


@pytest.mark.parametrize("helper_kind", ["func", "real_func"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_specializes_only_observed_static_dependencies(tmp_path, monkeypatch, helper_kind):
    import itertools

    if helper_kind == "real_func" and ti.cfg.arch == ti.vulkan:
        pytest.skip("Real functions require LLVM; the separate rejection test covers Vulkan")
    source = STATIC_DOMAIN_SOURCE
    if helper_kind == "real_func":
        source = source.replace("@ti.func\n    def mode(self):", "@ti.real_func\n    def mode(self) -> ti.i32:")
    package = _source_package(tmp_path, monkeypatch, source, initializer="from .model import Model, Equivalent\n")
    manifest = ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert manifest["version"] == 11
    families = manifest["entry_families"]
    assert sorted(len(f["variants"]) for f in families.values()) == [2, 2, 4, 4]
    assert len(manifest["entries"]) == 8  # 4 compute + 2 other + plain + unused public function
    assert all("unused" not in f["domain"] for f in families.values())
    branching = next(f for f in families.values() if len(f["variants"]) == 4)
    assert {tuple(sorted(v["selection"])) for v in branching["variants"]} == {("first", "second"), ("first", "third")}
    with zipfile.ZipFile(tmp_path / "sdk/code.tcm") as archive:
        code = [item for item in archive.infolist() if item.filename.endswith((".ll", ".bc", ".spv"))]
        assert len(code) == 8
    with Runtime(manifest["target"]) as runtime, load_package(runtime, tmp_path / "sdk") as loaded:
        a, b = loaded.host.Model(7, False, False, False, 1), loaded.host.Equivalent(19, True, True, True, 7)
        bound = a.compute
        for first, second, third in itertools.product((False, True), repeat=3):
            original = package.Model(7, first, second, third, 0)
            a.first, a.second, a.third = first, second, third
            original.compute()
            bound()
            np.testing.assert_array_equal(a.values.to_numpy(), original.values.to_numpy())
        b.compute()
        np.testing.assert_array_equal(b.values.to_numpy(), np.arange(19) + 10)
        a.first, a.second = True, False
        builder = loaded.module("definition_fixture.model").ti.graph.GraphBuilder()
        builder.dispatch(a.compute)
        graph = builder.compile()
        with graph.bind({}) as prepared:
            a.first = False
            prepared.run()
            np.testing.assert_array_equal(a.values.to_numpy(), np.arange(7) + 20)
        a.first = 1  # bool and int are distinct static identities
        with pytest.raises(ValueError, match="outside.*domain"):
            bound()
        resources = set(runtime._resources)
        with pytest.raises(ValueError, match="outside.*domain"):
            loaded.host.Model(3, False, False, False, 99)
        assert set(runtime._resources) == resources
    probe = tmp_path / "consume_static.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'definition_fixture','taichi','taichi_forge'}:
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, Block())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a,b=code.host.Model(3,False,True,False,0),code.host.Model(19,True,True,False,6)
            a.compute()
            b.compute()
            np.testing.assert_array_equal(a.values.to_numpy(),np.arange(3)+40)
            np.testing.assert_array_equal(b.values.to_numpy(),np.arange(19)+10)
            a.second=False
            a.other()
            np.testing.assert_array_equal(a.values.to_numpy(),-5)
        print('STATIC_DOMAINS_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STATIC_DOMAINS_OK" in result.stdout


@pytest.mark.parametrize("choices", ["()", "(False, False)", "(self.first,)", "(object(),)"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_unproven_static_domains(tmp_path, monkeypatch, choices):
    source = STATIC_DOMAIN_SOURCE.replace("'first', choices=(False, True)", f"'first', choices={choices}")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="Static domain"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_an_uncompiled_declared_static_branch(tmp_path, monkeypatch):
    source = STATIC_DOMAIN_SOURCE.replace("result = 10", "result = self.unqualified_parameter")
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match="undeclared capture self.unqualified_parameter"):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert sys.modules["definition_fixture.model"].constructors == 0
    assert not (tmp_path / "sdk").exists()


TARGET_FIELDS_SOURCE = """
import taichi_forge as ti
constructors = 0
@ti.data_oriented
class Model:
    def __init__(self, n: int, redirect=False):
        global constructors
        constructors += 1
        self.n = n
        scope = ti.experimental.ResourceScope(self)
        self.arch = ti.lang.impl.current_cfg().arch
        if redirect:
            self.redirect()
        if self.arch == ti.cuda:
            extent = scope.dim('n') * 2
            scope.a = scope.field(ti.f32, shape=extent)
        else:
            extent = tuple(scope.dim(name) + 1 for name in ('n',))
            scope.a = scope.field(ti.f32, shape=extent)

    def redirect(self):
        self.arch = ti.cpu if self.arch == ti.cuda else ti.cuda

    def backend_name(self):
        return self.arch.name

    @ti.kernel
    def step(self, gain: ti.f32):
        for i in self.a:
            self.a[i] = gain * (i + self.n)
"""


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_package_selects_backend_resource_branches_with_runtime_guards(tmp_path, monkeypatch):
    package = _source_package(tmp_path, monkeypatch, TARGET_FIELDS_SOURCE)
    manifest = ti.aot.export_package(package, tmp_path / "sdk", arch=ti.cfg.arch)
    assert manifest["version"] == 6
    assert sys.modules["definition_fixture.model"].constructors == 0
    selections = manifest["target_selections"]["definition_fixture.model"]
    assert len(selections) == 1 and selections[0]["taken"] == (ti.cfg.arch == ti.cuda)
    expected = package.Model(7)
    expected.step(2.0)
    probe = tmp_path / "consume_target.py"
    probe.write_text(
        textwrap.dedent("""
        import importlib.abc
        import sys
        import numpy as np
        class NoPrivateSource(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('taichi', 'taichi_forge', 'definition_fixture'):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, NoPrivateSource())
        from taichi_forge_runtime.aot import Runtime, load_package
        with Runtime(sys.argv[2]) as runtime, load_package(runtime, sys.argv[1]) as code:
            a, b = code.host.Model(7), code.host.Model(19)
            for model in (a, b):
                model.step(2.0)
                count = model.n * 2 if runtime.arch == 'cuda' else model.n + 1
                assert model.a.shape == (count,)
                assert model.backend_name() == runtime.arch
                np.testing.assert_array_equal(np.frombuffer(model.a.read(), np.float32), 2 * (np.arange(count) + model.n))
            registered = set(runtime._resources)
            try:
                code.host.Model(3, redirect=True)
            except ValueError as error:
                assert 'Host backend condition changed after export' in str(error)
            else:
                raise AssertionError('Changed backend branch silently selected another layout')
            assert runtime._resources == registered
        print('TARGET_FIELDS_SOURCE_FREE_OK')
    """),
        encoding="utf8",
    )
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[2] / "packaging/runtime/src"),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
    )
    result = subprocess.run(
        [sys.executable, "-B", str(probe), str(tmp_path / "sdk"), manifest["target"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TARGET_FIELDS_SOURCE_FREE_OK" in result.stdout
    count = 14 if ti.cfg.arch == ti.cuda else 8
    np.testing.assert_array_equal(expected.a.to_numpy(), 2 * (np.arange(count) + 7))


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_does_not_compile_other_backend_resources_or_select_model_branches(tmp_path, monkeypatch):
    source = TARGET_FIELDS_SOURCE.replace(
        "scope.a = scope.field(ti.f32, shape=extent)",
        "scope.a = scope.field(ti.f32, shape=extent, placement=None)",
        1,
    )
    package = _source_package(tmp_path, monkeypatch, source)
    ti.aot.export_package(package, tmp_path / "sdk")  # CUDA-only unsupported placement is absent.
    module = tmp_path / "definition_fixture/model.py"
    module.write_text(source.replace("if self.arch == ti.cuda:", "if n > 4:"), encoding="utf8")
    monkeypatch.delitem(sys.modules, "definition_fixture.model")
    importlib.import_module("definition_fixture.model")
    with pytest.raises(ti.aot.PackageExportError, match="unconditional supported resource declaration"):
        ti.aot.export_package(package, tmp_path / "model_branch")
    assert not (tmp_path / "model_branch").exists()


@pytest.mark.parametrize("shadow", ["frontend", "tuple_argument", "tuple_global"])
@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_package_rejects_shadowed_backend_and_dimension_names(tmp_path, monkeypatch, shadow):
    source = TARGET_FIELDS_SOURCE
    if shadow == "frontend":
        source = source.replace("redirect=False):", "redirect=False, ti=None):")
        message = "shadowed frontend name"
    else:
        source = (
            source.replace("redirect=False):", "redirect=False, tuple=None):")
            if shadow == "tuple_argument"
            else "tuple = list\n" + source
        )
        message = "Field shape requires declared dimensions"
    package = _source_package(tmp_path, monkeypatch, source)
    with pytest.raises(ti.aot.PackageExportError, match=message):
        ti.aot.export_package(package, tmp_path / "sdk")
    assert not (tmp_path / "sdk").exists()
