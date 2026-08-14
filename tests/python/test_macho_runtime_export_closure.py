import importlib.util
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "misc"
    / "generate_macho_runtime_export_closure.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "generate_macho_runtime_export_closure", _SCRIPT
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def test_macho_demangler_removes_only_platform_cxx_prefix(monkeypatch):
    monkeypatch.setattr(
        _MODULE.posix_support,
        "_demangle",
        lambda symbols: {symbol: f"demangled:{symbol}" for symbol in symbols},
    )

    assert _MODULE._demangle(["__ZN6taichi4lang7Program4syncEv", "_plain"]) == {
        "__ZN6taichi4lang7Program4syncEv": (
            "demangled:_ZN6taichi4lang7Program4syncEv"
        ),
        "_plain": "demangled:_plain",
    }


def test_macho_forbidden_audit_uses_unprefixed_c_names(monkeypatch):
    monkeypatch.setattr(
        _MODULE,
        "_demangle",
        lambda symbols: {symbol: symbol for symbol in symbols},
    )

    assert _MODULE.audit_forbidden_exports(
        {"_LLVMContextCreate", "_taichi_runtime_anchor"}
    ) == {"llvm": ["_LLVMContextCreate"]}


def test_macho_export_list_is_exact(tmp_path):
    output = tmp_path / "taichi_runtime.exports"
    exports = ["__ZN6taichi4lang7Program4syncEv", "_taichi_runtime_anchor"]

    _MODULE._write_export_list(output, exports)

    assert output.read_text(encoding="utf-8") == (
        "__ZN6taichi4lang7Program4syncEv\n_taichi_runtime_anchor\n"
    )
