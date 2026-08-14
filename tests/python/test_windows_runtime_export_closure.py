import importlib.util
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "misc"
    / "generate_windows_runtime_export_closure.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "generate_windows_runtime_export_closure", _SCRIPT
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def test_parse_dumpbin_undefined_ignores_demangled_annotation():
    decorated = "?launch@Kernel@lang@taichi@@QEAAXXZ"
    dumpbin = f"""
  138 00000000 UNDEF  notype       External     | __imp_{decorated} (public: void __cdecl taichi::lang::Kernel::launch(void))
  144 00000000 UNDEF  notype       WeakExternal | ??_EExceptionForPython@taichi@@UEAAPEAXI@Z
"""

    assert _MODULE.parse_dumpbin_undefined(dumpbin) == {f"__imp_{decorated}"}


def test_build_closure_uses_only_shim_reachable_runtime_symbols():
    required = "?launch@Kernel@lang@taichi@@QEAAXXZ"
    raw_symbols = {
        required,
        "?unused@Program@lang@taichi@@QEAAXXZ",
        "??_7Program@lang@taichi@@6B@",
    }

    exports, manifest = _MODULE.build_closure(
        raw_symbols,
        {f"__imp_{required}", "__imp_?python_only@@YAXXZ"},
    )

    assert exports == [required, "taichi_runtime_anchor"]
    assert manifest["raw_defined_symbol_count"] == 3
    assert manifest["shim_required_runtime_symbol_count"] == 1
    assert manifest["exported_symbol_count"] == 2
    assert manifest["dropped_raw_symbol_count"] == 2
    assert len(manifest["export_set_sha256"]) == 64


def test_export_closure_is_deterministic():
    raw_symbols = {"?b@taichi@@YAXXZ", "?a@taichi@@YAXXZ"}
    undefined = {"__imp_?a@taichi@@YAXXZ", "__imp_?b@taichi@@YAXXZ"}

    first = _MODULE.build_closure(raw_symbols, undefined)
    second = _MODULE.build_closure(
        set(reversed(sorted(raw_symbols))),
        set(reversed(sorted(undefined))),
    )

    assert first == second


def test_dll_audit_includes_implicit_exports_and_is_deterministic():
    requested = "?launch@Kernel@lang@taichi@@QEAAXXZ"
    manifest = {
        "exports": [requested, "taichi_runtime_anchor"],
        "configured_export_limit": 8,
    }
    actual = {
        "?explicit_api@taichi@@YAXXZ",
        "taichi_runtime_anchor",
        requested,
    }

    audited = _MODULE.add_dll_audit(manifest, actual)

    assert audited["dll_audited"] is True
    assert audited["actual_exported_symbol_count"] == 3
    assert audited["implicit_exported_symbol_count"] == 1
    assert audited["actual_exports"] == sorted(actual)
    assert len(audited["actual_export_set_sha256"]) == 64


def test_dll_audit_rejects_missing_and_excessive_exports():
    manifest = {
        "exports": ["required", "taichi_runtime_anchor"],
        "configured_export_limit": 2,
    }

    try:
        _MODULE.add_dll_audit(manifest, {"taichi_runtime_anchor"})
    except RuntimeError as exc:
        assert "missing requested exports" in str(exc)
    else:
        raise AssertionError("missing requested export was accepted")

    try:
        _MODULE.add_dll_audit(
            manifest, {"required", "taichi_runtime_anchor", "unexpected"}
        )
    except RuntimeError as exc:
        assert "safety limit" in str(exc)
    else:
        raise AssertionError("excessive DLL export set was accepted")


def test_forbidden_export_owner_ignores_third_party_signature_types():
    assert (
        _MODULE.forbidden_export_family(
            "?use@lang@taichi@@YAXPEAVType@llvm@@@Z",
            "void __cdecl taichi::lang::use(class llvm::Type *)",
        )
        is None
    )
    assert (
        _MODULE.forbidden_export_family(
            "?getInt32Ty@Type@llvm@@SAPEAV12@AEAVLLVMContext@2@@Z",
            "class llvm::Type * __cdecl llvm::Type::getInt32Ty("
            "class llvm::LLVMContext &)",
        )
        == "llvm"
    )


def test_dll_audit_rejects_implicit_third_party_definition_owner(monkeypatch):
    requested = "?launch@Kernel@lang@taichi@@QEAAXXZ"
    llvm_owned = "?getInt32Ty@Type@llvm@@SAPEAV12@AEAVLLVMContext@2@@Z"
    manifest = {
        "exports": [requested, "taichi_runtime_anchor"],
        "configured_export_limit": 8,
    }
    monkeypatch.setattr(
        _MODULE,
        "_undecorate",
        lambda symbol: (
            "llvm::Type::getInt32Ty(llvm::LLVMContext&)"
            if symbol == llvm_owned
            else symbol
        ),
    )

    try:
        _MODULE.add_dll_audit(
            manifest,
            {requested, "taichi_runtime_anchor", llvm_owned},
        )
    except RuntimeError as exc:
        assert "bundled third-party APIs" in str(exc)
    else:
        raise AssertionError("implicit third-party export was accepted")
