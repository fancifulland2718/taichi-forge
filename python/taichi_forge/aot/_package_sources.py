"""Source boundaries for the existing decorator-driven package compiler."""

import ast
from pathlib import Path
from pkgutil import get_data
from types import ModuleType

from taichi_forge.tools._host_inventory import audit_source_imports, source_inventory


def shader_assets(sources, modules, syntax=None):
    """Collect compiled shaders read by package Host code via pkgutil.get_data."""
    payloads = {}
    for name, source, relative, text in sources:
        if ".spv" not in text:
            continue
        aliases = {key for key, value in vars(modules[name]).items() if value is get_data}
        if not aliases:
            continue
        tree = syntax.get(name) if syntax is not None else None
        for call in ast.walk(tree if tree is not None else ast.parse(text.removeprefix("\ufeff"))):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id in aliases and len(call.args) == 2
                    and isinstance(call.args[0], ast.Name) and call.args[0].id == "__package__"
                    and isinstance(call.args[1], ast.Constant) and isinstance(call.args[1].value, str)):
                continue
            path = Path(call.args[1].value)
            if path.suffix != ".spv":
                continue
            resolved = (source.parent / path).resolve()
            if path.anchor or ".." in path.parts or not resolved.is_relative_to(source.parent.resolve()):
                raise ValueError(f"Shader asset outside package {name}: {path}")
            key = "host/" + (relative.parent / path).as_posix()
            data = resolved.read_bytes()
            if len(data) < 20 or len(data) % 4 or data[:4] != b"\x03\x02\x23\x07":
                raise ValueError(f"Invalid SPIR-V shader asset: {resolved}")
            payloads[key] = data
    return payloads


def package_roots(package, source_root):
    """Normalize public roots without inventing an aggregate application module."""
    packages = (package,) if isinstance(package, ModuleType) else package
    if not isinstance(packages, (tuple, list)) or not packages:
        raise TypeError("Package export requires an imported source module or a nonempty sequence of modules")
    roots = {}
    for item in packages:
        if not isinstance(item, ModuleType) or not getattr(item, "__file__", None):
            raise TypeError("Package export roots must be imported source modules or packages")
        previous = roots.get(item.__name__)
        if previous is not None and Path(previous.__file__).resolve() != Path(item.__file__).resolve():
            raise ValueError("Conflicting package root origins: " + item.__name__)
        roots.setdefault(item.__name__, item)
    if len(roots) > 1 and source_root is None:
        raise ValueError("Multiple package roots require a common application source_root")
    return tuple(roots.values())


def collect_sources(package, source_root, *, source_snapshots=None):
    packages = package_roots(package, source_root)
    package = packages[0]
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
    for item in packages:
        selected = available.get(item.__name__)
        if selected is None or Path(selected["source"]).resolve() != Path(item.__file__).resolve():
            raise ValueError("Selected module origin does not match the application source root: " + item.__name__)
    roots = sorted(
        {
            name
            for name in available
            for item in packages
            if name == item.__name__ or (hasattr(item, "__path__") and name.startswith(item.__name__ + "."))
        }
    )
    report = audit_source_imports(available, roots=roots, include_type_checking=False,
                                  source_snapshots=source_snapshots)
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
