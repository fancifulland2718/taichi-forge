"""Opt-in producer test; requires a working host C++ toolchain, not a GPU."""

import ast
import gc
import importlib.util
import os
from pathlib import Path
import weakref

import pytest

from taichi_forge.tools._native_host import NativeConstructors


@pytest.mark.skipif(os.environ.get("TI_TEST_NATIVE_HOST") != "1", reason="requires the SDK host C++ toolchain")
def test_native_template_semantics_ownership_and_errors(tmp_path, monkeypatch):
    from setuptools import Distribution, Extension
    from setuptools.command.build_ext import build_ext

    function = ast.parse('''
def literal_template(x):
    """A template with source locations and mixed Python constants."""
    payload = ("unicode\\u4e2d\\ud800\\x00", b"bytes\\x00\\xff", True, False, None, ...,
               123456789012345678901234567890, 1e309, -0.0, 1.5j)
    return x[1:3] if x else payload
''').body[0]
    emitter = NativeConstructors("native_test")
    emitter.constructor(function, "unused_python_wrapper")
    _, cpp, _ = emitter.emit("unused.h")
    cpp += f"""
static PyObject *build(PyObject *, PyObject *args) {{
  int index;
  if (!PyArg_ParseTuple(args, "i", &index)) return nullptr;
  return {emitter.symbol}(index);
}}
static PyMethodDef methods[] = {{ {{"build", build, METH_VARARGS, nullptr}}, {{nullptr, nullptr, 0, nullptr}} }};
static PyModuleDef module = {{ PyModuleDef_HEAD_INIT, "forge_native_template_probe", nullptr, -1, methods }};
PyMODINIT_FUNC PyInit_forge_native_template_probe() {{ return PyModule_Create(&module); }}
"""
    source = tmp_path / "templates.cpp"
    source.write_text(cpp, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    extension = Extension(
        "forge_native_template_probe",
        [source.name],
        language="c++",
        extra_compile_args=["/EHsc", "/utf-8"] if os.name == "nt" else ["-std=c++17"],
    )
    command = build_ext(Distribution({"name": "native-template-test", "ext_modules": [extension]}))
    command.build_lib, command.build_temp = "lib", "objects"
    command.ensure_finalized()
    command.run()
    binary = Path(command.get_ext_fullpath(extension.name))
    spec = importlib.util.spec_from_file_location(extension.name, binary)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    expected = ast.dump(ast.Module(body=[function], type_ignores=[]), include_attributes=True)
    references = []
    for _ in range(3):
        tree = native.build(0)
        assert ast.dump(tree, include_attributes=True) == expected
        references.extend(weakref.ref(node) for node in ast.walk(tree))
        del tree
    gc.collect()
    assert all(ref() is None for ref in references)
    with pytest.raises(IndexError, match="Unknown compiled template"):
        native.build(1)

    def fail(*args, **kwargs):
        raise ValueError("injected constructor failure")

    with monkeypatch.context() as patch:
        patch.setattr(ast, "Name", fail)
        with pytest.raises(ValueError, match="injected constructor failure"):
            native.build(0)
    assert ast.dump(native.build(0), include_attributes=True) == expected
