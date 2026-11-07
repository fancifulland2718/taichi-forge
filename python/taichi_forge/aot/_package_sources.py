"""Source boundaries for the existing decorator-driven package compiler."""

from pathlib import Path

from taichi_forge.tools._host_inventory import audit_source_imports, source_inventory


def collect_sources(package, source_root):
    source = Path(package.__file__).resolve()
    if source_root is None:
        result = {}
        paths = sorted(source.parent.rglob("*.py")) if hasattr(package, "__path__") else [source]
        for path in paths:
            relative = path.relative_to(source.parent) if hasattr(package, "__path__") else Path("__init__.py")
            parts = relative.with_suffix("").parts
            suffix = parts[:-1] if parts[-1] == "__init__" else parts
            result[".".join((package.__name__, *suffix))] = (path, relative)
        return result, (), None

    # Applications may contain a ``taichi`` compatibility shim that re-exports
    # Forge. Numerical frontend/runtime implementations remain external; their
    # source is never another application compilation unit.
    external_runtime = {"taichi", "taichi_forge", "taichi_forge_runtime"}
    available = {
        name: item for name, item in source_inventory(source_root).items() if name.split(".")[0] not in external_runtime
    }
    selected = available.get(package.__name__)
    if selected is None or Path(selected["source"]).resolve() != source:
        raise ValueError("Selected module origin does not match the application source root")
    roots = [
        name
        for name in available
        if name == package.__name__ or (hasattr(package, "__path__") and name.startswith(package.__name__ + "."))
    ]
    report = audit_source_imports(available, roots=roots)
    if report["unresolved_local_imports"]:
        raise ValueError("Unresolved local source imports: " + repr(report["unresolved_local_imports"]))
    result = {name: (Path(available[name]["source"]), Path(available[name]["relative"])) for name in report["visited"]}
    parents = set()
    for name in result:
        parent = name.rpartition(".")[0]
        while parent:
            parents.add(parent)
            parent = parent.rpartition(".")[0]
    namespaces = sorted(parents - result.keys())
    return result, namespaces, report
