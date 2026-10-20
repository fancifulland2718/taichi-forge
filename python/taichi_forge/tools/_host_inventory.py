"""Read-only static import audit for an explicit source-hidden host inventory.

Run as a producer script without importing the DSL or application. Findings are
review candidates, not a proof of dynamic reachability or an automatic manifest.
"""

import argparse
import ast
from collections import deque
import hashlib
import importlib.util
import json
from pathlib import Path

if __package__:
    from ._host_build import read_spec
else:
    from _host_build import read_spec


def audit_imports(filename, *, source_root=None, roots=()):
    frozen_root, selected, aliases, _ = read_spec(filename)
    candidate_root = Path(source_root).resolve() if source_root is not None else frozen_root
    if not candidate_root.is_dir():
        raise ValueError(f"Missing candidate source root: {candidate_root}")
    candidates = {}
    for path in sorted(candidate_root.rglob("*.py")):
        relative = path.relative_to(candidate_root)
        if not path.resolve().is_relative_to(candidate_root):
            raise ValueError(f"Out-of-root candidate source: {relative}")
        parts = relative.with_suffix("").parts
        package = parts[-1] == "__init__"
        name = ".".join(parts[:-1] if package else parts)
        if not name or not all(part.isidentifier() for part in name.split(".")):
            continue
        if name in candidates:
            raise ValueError(f"Ambiguous module/package candidate: {name}")
        # Exact filename spelling matters even on case-insensitive filesystems:
        # a class named Coupler does not identify a file named coupler.py.
        candidates[name] = dict(source=str(path), relative=relative.as_posix(), package=package)
    available = dict(candidates, **selected)
    local_roots = {name.split(".")[0] for name in available} | {name.split(".")[0] for name in aliases}
    namespace_prefixes = set()
    for name in available:
        parent = name.rpartition(".")[0]
        while parent:
            namespace_prefixes.add(parent)
            parent = parent.rpartition(".")[0]
    alias_names = sorted(aliases, key=len, reverse=True)

    def resolve(name):
        for alias in alias_names:
            if name == alias or name.startswith(alias + "."):
                contract = aliases[alias]
                return contract["target"] + name[len(alias) :], contract["registered_by"]
        return name, None

    roots = tuple(sorted(set(roots or selected)))
    for name in roots:
        target, _ = resolve(name)
        if target not in available:
            raise ValueError(f"Audit root is not an inventoried local module: {name}")
    pending = deque((name, None, 0, "root") for name in roots)
    seen, edges, external, unresolved, dynamic, wildcard = set(), [], set(), [], [], []
    missing = {}
    while pending:
        requested, importer, line, kind = pending.popleft()
        name, registrar = resolve(requested)
        if registrar is not None:
            pending.append((registrar, importer, line, "alias_registrar"))
        if name not in available:
            if name.split(".")[0] in local_roots:
                # A namespace package has no initializer to compile.
                if name not in namespace_prefixes:
                    unresolved.append(dict(module=name, importer=importer, line=line))
            else:
                external.add(name)
            continue
        edges.append(dict(module=name, requested=requested, importer=importer, line=line, kind=kind))
        if name in seen:
            continue
        seen.add(name)
        item = available[name]
        data = Path(item["source"]).read_bytes()
        if name not in selected:
            missing[name] = dict(path=item["relative"], sha256=hashlib.sha256(data).hexdigest())
        parent = name.rpartition(".")[0]
        while parent:
            if parent in available and available[parent]["package"]:
                pending.append((parent, name, 0, "package_initializer"))
            parent = parent.rpartition(".")[0]
        package = name if item["package"] else name.rpartition(".")[0]
        for node in ast.walk(ast.parse(data, filename=item["source"])):
            if isinstance(node, ast.Import):
                pending.extend((entry.name, name, node.lineno, "import") for entry in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = (
                    importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                    if node.level
                    else node.module
                )
                if not base or base == "__future__":
                    continue
                pending.append((base, name, node.lineno, "from"))
                for entry in node.names:
                    if entry.name == "*":
                        wildcard.append(dict(module=name, line=node.lineno, imported=base))
                    else:
                        child = base + "." + entry.name
                        if resolve(child)[0] in available:
                            pending.append((child, name, node.lineno, "from_member"))
            elif isinstance(node, ast.Call):
                called = node.func
                if (isinstance(called, ast.Name) and called.id in ("__import__", "import_module")) or (
                    isinstance(called, ast.Attribute) and called.attr == "import_module"
                ):
                    dynamic.append(dict(module=name, line=node.lineno))
    return dict(
        version=1,
        coverage="Static syntax only; optional branches may over-report and dynamic imports may be missed.",
        roots=list(roots),
        visited=sorted(seen),
        missing_modules=dict(sorted(missing.items())),
        unresolved_local_imports=unresolved,
        external_imports=sorted(external),
        import_edges=edges,
        dynamic_import_sites=dynamic,
        wildcard_import_sites=wildcard,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--source-root", help="Complete candidate source tree; selected modules use frozen spec bytes")
    parser.add_argument(
        "--root", action="append", default=[], help="Public entry module; default: all selected modules"
    )
    parser.add_argument(
        "--fail-on-missing", action="store_true", help="Return 1 for missing or unresolved local imports"
    )
    args = parser.parse_args(argv)
    report = audit_imports(args.spec, source_root=args.source_root, roots=args.root)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return int(args.fail_on_missing and bool(report["missing_modules"] or report["unresolved_local_imports"]))


if __name__ == "__main__":
    raise SystemExit(main())
