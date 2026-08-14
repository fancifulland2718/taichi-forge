import importlib.util
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "misc"
    / "generate_elf_runtime_export_closure.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "generate_elf_runtime_export_closure", _SCRIPT
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def test_parse_posix_nm_symbols_handles_objects_and_archives():
    text = """
taichi_symbol T 10 20
libcore.a[file.o]: llvm_symbol W 30 40
undefined_symbol U
T T 50 60
"""

    assert _MODULE.parse_posix_nm_symbols(text) == {
        "taichi_symbol",
        "llvm_symbol",
        "undefined_symbol",
        "T",
    }


def test_strip_elf_symbol_version_preserves_mangled_name():
    assert (
        _MODULE.strip_elf_symbol_version(
            "_ZN6taichi4lang7Program4syncEv@@TAICHI_FORGE_RUNTIME_PRIVATE_1"
        )
        == "_ZN6taichi4lang7Program4syncEv"
    )


def test_forbidden_export_audit_uses_definition_owner_not_signature_types(
    monkeypatch,
):
    owned_by_taichi = "_ZN6taichi4lang3useEPN4llvm4TypeE"
    owned_by_llvm = "_ZN4llvm4Type9getInt32TyERNS_11LLVMContextE"

    monkeypatch.setattr(
        _MODULE,
        "_demangle",
        lambda symbols: {
            owned_by_taichi: "taichi::lang::use(llvm::Type*)",
            owned_by_llvm: "llvm::Type::getInt32Ty(llvm::LLVMContext&)",
        },
    )

    assert _MODULE.audit_forbidden_exports(
        {owned_by_taichi, owned_by_llvm}
    ) == {"llvm": [owned_by_llvm]}


def test_version_script_is_exact_and_localizes_everything_else(tmp_path):
    path = tmp_path / "taichi_runtime.map"
    exports = ["_ZN6taichi4lang7Program4syncEv", "taichi_runtime_anchor"]

    _MODULE._write_version_script(path, exports)

    assert path.read_text(encoding="utf-8") == (
        "TAICHI_FORGE_RUNTIME_PRIVATE_1 {\n"
        "  global:\n"
        "    _ZN6taichi4lang7Program4syncEv;\n"
        "    taichi_runtime_anchor;\n"
        "  local:\n"
        "    *;\n"
        "};\n"
    )


def test_common_closure_counts_direct_and_shared_odr_once():
    exports, manifest = _MODULE.build_export_closure(
        {"direct", "shared", "unused"},
        {"direct", "shared"},
        platform="linux-elf",
        additional_required_symbols={"shared"},
    )

    assert exports == ["direct", "shared", "taichi_runtime_anchor"]
    assert manifest["shim_direct_runtime_symbol_count"] == 2
    assert manifest["shim_shared_odr_symbol_count"] == 0
    assert manifest["shim_required_runtime_symbol_count"] == 2

    exports, manifest = _MODULE.build_export_closure(
        {"direct", "shared", "unused"},
        {"direct"},
        platform="linux-elf",
        additional_required_symbols={"shared"},
    )
    assert exports == ["direct", "shared", "taichi_runtime_anchor"]
    assert manifest["shim_direct_runtime_symbol_count"] == 1
    assert manifest["shim_shared_odr_symbol_count"] == 1
    assert manifest["shim_required_runtime_symbol_count"] == 2


def test_collision_probes_prioritize_high_level_taichi_owners():
    symbols = ["anchor", "program", "kernel", "unrelated"]
    demangled = {
        "program": "taichi::lang::Program::sync()",
        "kernel": "taichi::lang::Kernel::launch()",
        "unrelated": "taichi::lang::Type::is_integral()",
    }

    assert _MODULE.select_private_abi_collision_probes(
        symbols,
        demangled,
        anchor="anchor",
        limit=3,
    ) == ["anchor", "kernel", "program"]
