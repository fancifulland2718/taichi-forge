"""Producer-only native constructors for compiled application DSL templates.

These functions emit build input. Neither this C++ source nor the intermediate
Cython source belongs in an application distribution. The extension shares the
same Python frontend protocol as the ordinary compiled-template producer.
"""

import ast
import hashlib
import math

_SUPPORT = r"""
#include <Python.h>
#include <initializer_list>
#include <limits>
#include <new>
#include <vector>

namespace {
#if defined(_MSC_VER)
#define FORGE_HOST_NOINLINE __declspec(noinline)
#else
#define FORGE_HOST_NOINLINE __attribute__((noinline))
#endif
struct PythonFailure {};
class Builder {
 public:
  Builder() : ast_(own(PyImport_ImportModule("ast"))) {}
  ~Builder() { for (auto it = owned_.rbegin(); it != owned_.rend(); ++it) Py_DECREF(*it); }
  FORGE_HOST_NOINLINE PyObject *own(PyObject *value) {
    if (!value) throw PythonFailure{};
    try { owned_.push_back(value); }
    catch (...) { Py_DECREF(value); throw; }
    return value;
  }
  FORGE_HOST_NOINLINE PyObject *ref(PyObject *value) { Py_INCREF(value); return own(value); }
  FORGE_HOST_NOINLINE PyObject *integer(const char *value) {
    return own(PyLong_FromString(const_cast<char *>(value), nullptr, 10));
  }
  FORGE_HOST_NOINLINE PyObject *text(const char *value, Py_ssize_t size) {
    return own(PyUnicode_DecodeUTF8(value, size, "surrogatepass"));
  }
  FORGE_HOST_NOINLINE PyObject *bytes(const char *value, Py_ssize_t size) {
    return own(PyBytes_FromStringAndSize(value, size));
  }
  FORGE_HOST_NOINLINE PyObject *tuple(std::initializer_list<PyObject *> values) {
    auto *result = own(PyTuple_New(values.size()));
    Py_ssize_t index = 0;
    for (auto *value : values) { Py_INCREF(value); PyTuple_SET_ITEM(result, index++, value); }
    return result;
  }
  FORGE_HOST_NOINLINE PyObject *list(std::initializer_list<PyObject *> values) {
    auto *result = own(PyList_New(values.size()));
    Py_ssize_t index = 0;
    for (auto *value : values) { Py_INCREF(value); PyList_SET_ITEM(result, index++, value); }
    return result;
  }
  FORGE_HOST_NOINLINE PyObject *node(const char *kind, std::initializer_list<PyObject *> fields) {
    auto *type = own(PyObject_GetAttrString(ast_, kind));
    return own(PyObject_CallObject(type, tuple(fields)));
  }
  FORGE_HOST_NOINLINE void location(PyObject *node, long line, long column, long end_line, long end_column) {
    const char *names[] = {"lineno", "col_offset", "end_lineno", "end_col_offset"};
    const long values[] = {line, column, end_line, end_column};
    for (int i = 0; i < 4; ++i) {
      auto *value = values[i] < 0 ? ref(Py_None) : own(PyLong_FromLong(values[i]));
      if (PyObject_SetAttrString(node, names[i], value) < 0) throw PythonFailure{};
    }
  }
  PyObject *result(PyObject *value) { Py_INCREF(value); return value; }
 private:
  std::vector<PyObject *> owned_;
  PyObject *ast_;
};
"""


def _bytes(value):
    # Separate adjacent literals so a following hexadecimal digit cannot extend
    # an escape; embedded NUL and non-ASCII values preserve their exact length.
    return '"' + '""'.join("\\x%02x" % byte for byte in value) + '"'


def _float(value):
    if math.isnan(value):
        return "std::numeric_limits<double>::quiet_NaN()"
    if math.isinf(value):
        return ("-" if value < 0 else "") + "std::numeric_limits<double>::infinity()"
    return repr(value)


class NativeConstructors:
    """Collect executable template constructors while lowering one host module."""

    def __init__(self, module_name):
        self.symbol = "forge_host_template_" + hashlib.sha256(module_name.encode()).hexdigest()[:16]
        self.functions = []

    def constructor(self, function, name):
        statements = []

        def assign(expression):
            variable = "n" + str(len(statements))
            statements.append(f"  auto *{variable} = {expression};")
            return variable

        def encode(value):
            if isinstance(value, ast.AST):
                fields = ", ".join(encode(item) for _, item in ast.iter_fields(value))
                result = assign(f'b.node("{type(value).__name__}", {{{fields}}})')
                if hasattr(value, "lineno"):
                    end_line, end_col = getattr(value, "end_lineno", None), getattr(value, "end_col_offset", None)
                    position = (
                        value.lineno,
                        value.col_offset,
                        -1 if end_line is None else end_line,
                        -1 if end_col is None else end_col,
                    )
                    statements.append(f"  b.location({result}, {', '.join(map(str, position))});")
                return result
            if isinstance(value, (list, tuple)):
                fields = ", ".join(map(encode, value))
                return assign(f"b.{type(value).__name__}({{{fields}}})")
            if value is None or value is Ellipsis:
                return assign("b.ref(" + ("Py_None" if value is None else "Py_Ellipsis") + ")")
            if isinstance(value, bool):
                return assign("b.ref(" + ("Py_True" if value else "Py_False") + ")")
            if isinstance(value, int):
                return assign(f'b.integer("{value}")')
            if isinstance(value, float):
                return assign(f"b.own(PyFloat_FromDouble({_float(value)}))")
            if isinstance(value, complex):
                return assign(f"b.own(PyComplex_FromDoubles({_float(value.real)}, {_float(value.imag)}))")
            if isinstance(value, (str, bytes)):
                raw = value.encode("utf-8", "surrogatepass") if isinstance(value, str) else value
                kind = "text" if isinstance(value, str) else "bytes"
                return assign(f"b.{kind}({_bytes(raw)}, {len(raw)})")
            raise TypeError(f"Unsupported compiled template value: {type(value).__name__}")

        result = encode(ast.Module(body=[function], type_ignores=[]))
        index = len(self.functions)
        self.functions.append(
            "static PyObject *build_"
            + str(index)
            + "(Builder &b) {\n"
            + "\n".join(statements)
            + f"\n  return {result};\n}}\n"
        )
        return ast.parse(f"def {name}():\n    return _forge_host_native({index})\n").body[0]

    def emit(self, header_name):
        """Return a Cython declaration, C++ source, and matching private header."""
        if not self.functions:
            return "", "", ""
        declaration = f'\ncdef extern from "{header_name}":\n    object _forge_host_native "{self.symbol}"(int index)\n'
        header = f'#pragma once\n#include <Python.h>\nextern "C" PyObject *{self.symbol}(int index);\n'
        dispatch = "\n".join(f"      case {i}: return b.result(build_{i}(b));" for i in range(len(self.functions)))
        cpp = _SUPPORT + "\n".join(self.functions) + f"""
}}  // namespace
extern "C" PyObject *{self.symbol}(int index) {{
  try {{
    Builder b;
    switch (index) {{
{dispatch}
      default: PyErr_SetString(PyExc_IndexError, "Unknown compiled template"); return nullptr;
    }}
  }} catch (const PythonFailure &) {{ return nullptr; }}
    catch (const std::bad_alloc &) {{ return PyErr_NoMemory(); }}
    catch (...) {{ PyErr_SetString(PyExc_RuntimeError, "Compiled template construction failed"); return nullptr; }}
}}
"""
        return declaration, cpp, header
