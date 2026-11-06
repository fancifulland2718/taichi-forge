"""Experimental definition-driven package export; no model execution tracing.

The first supported slice is scalar/ndarray owners with ordinary decorators.
Existing component lowering is an internal compiler service, not a registration
or recording API required of the application. Unsupported definitions fail the
whole export rather than silently disappearing from the package.
"""

import ast
import copy
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import shutil
import tempfile
from types import ModuleType

from taichi_forge.tools._compiled_host import decorator_resolver

_DTYPES = frozenset(("f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"))
_DECORATORS = frozenset(("kernel", "func", "real_func", "pyfunc", "data_oriented"))
_PREFIX = "_forge_package_"


class PackageExportError(ValueError):
    """A definition cannot yet be represented by the package exporter."""


def _fail(module, node, message):
    raise PackageExportError(f"{module}:{getattr(node, 'lineno', 1)}: {message}")


def _owner_schema(module, node, kind):
    """Infer types from syntax, never run __init__ or select a model branch."""
    if node.bases or node.keywords:
        _fail(module, node, "inherited owners are not yet supported")
    constructor = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
    if constructor is None:
        return {}
    annotations = {
        a.arg: a.annotation
        for a in (*constructor.args.posonlyargs, *constructor.args.args, *constructor.args.kwonlyargs)
    }
    schema = {}
    for statement in ast.walk(constructor):
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        for target in targets:
            if not (
                isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"
            ):
                continue
            value = statement.value
            if isinstance(value, ast.Call) and kind(value.func) == "ndarray":
                if statement not in constructor.body or len(targets) != 1:
                    _fail(module, statement, "conditional or aliased allocation requires a resource contract")
                arguments = dict(zip(("dtype", "shape"), value.args))
                if len(value.args) > 2 or any(k.arg not in ("dtype", "shape") for k in value.keywords):
                    _fail(module, value, "only scalar ndarray dtype and shape are supported")
                arguments.update((k.arg, k.value) for k in value.keywords)
                dtype = kind(arguments.get("dtype"))
                shape = arguments.get("shape")
                if dtype not in _DTYPES or shape is None:
                    _fail(module, value, "ndarray requires a fixed scalar dtype and explicit shape")
                rank = len(shape.elts) if isinstance(shape, (ast.Tuple, ast.List)) else 1
                spec = dict(kind="ndarray", dtype=dtype, ndim=rank)
            else:
                annotation = annotations.get(value.id) if isinstance(value, ast.Name) else None
                dtype = {"int": "i32", "float": "f32", "bool": "i32"}.get(
                    annotation.id if isinstance(annotation, ast.Name) else None
                )
                dtype = dtype or (kind(annotation) if annotation is not None else None)
                if isinstance(value, ast.Constant):
                    dtype = {int: "i32", float: "f32", bool: "i32"}.get(type(value.value))
                if dtype not in _DTYPES:
                    continue  # Host-only data is preserved; a numerical use diagnoses missing type information.
                spec = dict(kind="value", dtype=dtype)
            if target.attr in schema and schema[target.attr] != spec:
                _fail(module, statement, f"attribute {target.attr!r} has conflicting types")
            schema[target.attr] = spec
    return schema


def _analyze(module, source):
    tree = ast.parse(source)
    if any(isinstance(n, ast.Name) and n.id.startswith(_PREFIX) for n in ast.walk(tree)):
        _fail(module, tree, "reserved package-export identifier")
    kind = decorator_resolver(tree)
    definitions, owners = [], {}
    supported = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            decorators = [kind(d) for d in node.decorator_list]
            if "data_oriented" not in decorators:
                continue
            if decorators != ["data_oriented"]:
                _fail(module, node, "owner decorator combinations are not yet supported")
            owners[node.name] = _owner_schema(module, node, kind)
            members = [(node.name, n) for n in node.body]
        else:
            members = [(None, node)]
        for owner, entry in members:
            if not isinstance(entry, ast.FunctionDef):
                continue
            decorators = [kind(d) for d in entry.decorator_list]
            dsl = next((d for d in decorators if d in _DECORATORS), None)
            if dsl is None:
                continue
            if dsl not in ("kernel", "func") or decorators != [dsl]:
                _fail(module, entry, "only ordinary kernel/func methods are supported in this prototype")
            supported.add(id(entry))
            definitions.append(
                dict(owner=owner, node=entry, kind=dsl, name=".".join(p for p in (owner, entry.name) if p))
            )
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(kind(d) in _DECORATORS for d in node.decorator_list)
            and id(node) not in supported
        ):
            _fail(module, node, "nested, inherited, or asynchronous numerical definitions require further support")
    return tree, kind, owners, definitions


def _host_projection(module, tree, kind, owners, definitions, entries, package_name, device_functions, is_package):
    numerical = {item["name"]: item for item in definitions}
    definitions_at = {(item["node"].lineno, item["node"].col_offset): item for item in definitions}
    owners_at = {
        (node.lineno, node.col_offset) for node in tree.body if isinstance(node, ast.ClassDef) and node.name in owners
    }
    imported_helpers = set()

    def import_origin(node):
        if node.level:
            parent = module if is_package else module.rpartition(".")[0]
            return importlib.util.resolve_name("." * node.level + (node.module or ""), parent)
        return node.module

    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            origin = import_origin(node)
            imported_helpers.update(a.asname or a.name for a in node.names if f"{origin}.{a.name}" in device_functions)

    class Project(ast.NodeTransformer):
        owner = None

        def visit_Import(self, node):
            replacements = []
            for alias in node.names:
                if alias.name in ("taichi", "taichi_forge"):
                    replacements.append(
                        ast.Assign(
                            [ast.Name(alias.asname or alias.name, ast.Store())],
                            ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "host", ast.Load()),
                        )
                    )
                elif alias.name.startswith(("taichi.", "taichi_forge.")):
                    _fail(module, node, "frontend submodule imports require further Host projection support")
                elif alias.name == package_name or alias.name.startswith(package_name + "."):
                    _fail(module, node, "use relative package imports in the initial exporter")
                else:
                    replacements.append(ast.Import([alias]))
            return replacements

        def visit_ImportFrom(self, node):
            origin = import_origin(node)
            node.names = [a for a in node.names if f"{origin}.{a.name}" not in device_functions]
            if not node.names:
                return None
            if not node.level and node.module in ("taichi", "taichi_forge"):
                names = [a for a in node.names if a.name not in _DECORATORS]
                if any(a.name not in _DTYPES | {"ndarray"} for a in names):
                    _fail(module, node, "unsupported frontend dependency in Host code")
                return [
                    ast.Assign(
                        [ast.Name(a.asname or a.name, ast.Store())],
                        ast.Attribute(
                            ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "host", ast.Load()),
                            a.name,
                            ast.Load(),
                        ),
                    )
                    for a in names
                ]
            if not node.level and (node.module or "").startswith(("taichi.", "taichi_forge.")):
                _fail(module, node, "frontend submodule imports require further Host projection support")
            if not node.level and (node.module == package_name or (node.module or "").startswith(package_name + ".")):
                _fail(module, node, "use relative package imports in the initial exporter")
            return node

        def visit_ClassDef(self, node):
            previous, self.owner = self.owner, node.name
            managed = (node.lineno, node.col_offset) in owners_at
            if managed:
                node.decorator_list = []
            node = self.generic_visit(node)
            self.owner = previous
            if not node.body:
                node.body = [ast.Pass()]
            if managed:
                node.decorator_list = [
                    ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "managed_class", ast.Load())
                ]
            return node

        def visit_FunctionDef(self, node):
            definition = definitions_at.get((node.lineno, node.col_offset))
            if definition is None:
                return self.generic_visit(node)
            if definition["kind"] == "func":
                return None
            descriptor = ast.Call(
                ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "entry", ast.Load()),
                [ast.Constant(entries[definition["name"]]["symbol"])],
                [],
            )
            return ast.copy_location(ast.Assign([ast.Name(node.name, ast.Store())], descriptor), node)

        def visit_Attribute(self, node):
            if (
                isinstance(node.ctx, ast.Load)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and numerical.get(f"{self.owner}.{node.attr}", {}).get("kind") == "func"
            ):
                _fail(module, node, "Host references a device-only helper")
            role = kind(node)
            if role is not None and role not in _DTYPES | {"ndarray"}:
                _fail(module, node, f"Host frontend operation {role!r} needs a Runtime projection")
            return self.generic_visit(node)

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and (
                numerical.get(node.id, {}).get("kind") == "func" or node.id in imported_helpers
            ):
                _fail(module, node, "Host references a device-only helper")
            return node

        def visit_Call(self, node):
            if kind(node) in _DTYPES:
                _fail(module, node, "Host scalar casts require further Runtime projection support")
            return self.generic_visit(node)

    projected = Project().visit(copy.deepcopy(tree))
    # Preserve future-import placement and the original module docstring.
    position = 0
    for node in projected.body:
        if (isinstance(node, ast.ImportFrom) and node.module == "__future__") or (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        ):
            position += 1
        else:
            break
    projected.body[position:position] = ast.parse(
        f"from taichi_forge_runtime.aot._package import bind_module as {_PREFIX}bind\n"
        f"{_PREFIX}runtime = {_PREFIX}bind(__name__)\n"
    ).body
    ast.fix_missing_locations(projected)
    compile(projected, f"<projected-host:{module}>", "exec")
    return ast.unparse(projected) + "\n"


def export_package(package, output):
    """Experimentally export all decorated definitions in an imported module/package.

    Uses the currently initialized backend. No constructor, test, Host branch,
    initializer or step is executed to discover numerical coverage. Scalar
    ndarrays and directly typed scalar owner attributes are the initial scope.
    """
    import taichi_forge as ti
    from taichi_forge.aot._component import Array, Component, RuntimeValue
    from taichi_forge.aot.module import Module
    from taichi_forge.types.ndarray_type import NdarrayType

    if not isinstance(package, ModuleType) or not getattr(package, "__file__", None):
        raise TypeError("export_package requires an imported source module or package")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    source = Path(package.__file__).resolve()
    paths = sorted(source.parent.rglob("*.py")) if hasattr(package, "__path__") else [source]
    sources = []
    for path in paths:
        relative = path.relative_to(source.parent) if hasattr(package, "__path__") else Path("__init__.py")
        suffix = relative.with_suffix("").parts
        suffix = suffix[:-1] if suffix[-1] == "__init__" else suffix
        name = ".".join((package.__name__, *suffix))
        text = path.read_text(encoding="utf8")
        tree, kind, owners, definitions = _analyze(name, text)
        sources.append((name, path, relative, text, tree, kind, owners, definitions))
    module = Module()
    manifest = dict(
        version=1,
        kind="forge_definition_package_prototype",
        target=module._arch.name,
        discovery="all_decorated_definitions",
        entries={},
        files={},
        sources={},
        owners={},
    )
    payloads = {}
    device_functions = {
        name + "." + definition["name"]
        for name, _, _, _, _, _, _, definitions in sources
        for definition in definitions
        if definition["kind"] == "func"
    }
    for name, path, relative, text, tree, kind, schemas, definitions in sources:
        imported = package if name == package.__name__ else importlib.import_module(name)
        if Path(imported.__file__).resolve() != path:
            raise PackageExportError(f"Module origin differs from scanned source: {name}")
        owners = {}
        environments = {}
        for class_name, schema in schemas.items():
            cls = getattr(imported, class_name)
            owner = object.__new__(cls)  # Deliberately never call application __init__.
            declarations = {}
            for attribute, item in schema.items():
                dtype = getattr(ti, item["dtype"])
                if item["kind"] == "ndarray":
                    declarations[attribute] = Array(dtype, item["ndim"])
                    # Structural ABI witness only: every extent remains a runtime parameter.
                    value = ti.ndarray(dtype, (1,) * item["ndim"])
                else:
                    declarations[attribute] = RuntimeValue(dtype)
                    value = 0.0 if item["dtype"].startswith("f") else 0
                setattr(owner, attribute, value)
            owners[class_name], environments[class_name] = owner, Component(**declarations)
            manifest["owners"][name + "." + class_name] = schema
        entries = {}
        for definition in definitions:
            if definition["kind"] != "kernel":
                continue
            owner_name, node = definition["owner"], definition["node"]
            owner = owners.get(owner_name)
            entry = getattr(owner if owner is not None else imported, node.name)
            kernel = getattr(entry, "_primal", None)
            if kernel is None:
                _fail(name, node, "decorated definition did not resolve to a Forge kernel")
            arguments = kernel.arguments[1:] if owner is not None else kernel.arguments
            values, argument_schema = [], []
            for argument in arguments:
                dtype = str(argument.annotation)
                if isinstance(argument.annotation, NdarrayType):
                    annotation = argument.annotation
                    dtype = str(annotation.dtype)
                    if dtype not in _DTYPES or type(annotation.ndim) is not int or annotation.needs_grad:
                        _fail(name, node, "ndarray arguments require explicit scalar dtype/rank and no gradients")
                    values.append(ti.ndarray(annotation.dtype, (1,) * annotation.ndim))
                    argument_schema.append(dict(name=argument.name, kind="ndarray", dtype=dtype, ndim=annotation.ndim))
                elif dtype not in _DTYPES:
                    _fail(name, node, f"argument {argument.name!r} requires a supported scalar type; got {dtype}")
                else:
                    values.append(0.0 if dtype.startswith("f") else 0)
                    argument_schema.append(dict(name=argument.name, kind="value", dtype=dtype))
                if argument.default is not inspect.Parameter.empty:
                    _fail(name, node, "kernel default arguments are not yet supported")
            environment = environments[owner_name] if owner_name is not None else Component()
            try:
                code = environment.compile(entry, *values)
            except Exception as error:
                raise PackageExportError(f"{name}.{definition['name']}: {error}") from error
            symbol = "definition_" + hashlib.sha256((name + "." + definition["name"]).encode()).hexdigest()
            code.export(module, symbol)
            capture = [
                {"name": capture_name, "kind": schema[0], "dtype": schema[1]}
                for capture_name, schema in code.contract["captures"]
            ]
            # Ndarray descriptors are passed unchanged, not encoded as scalar arguments.
            for item in capture:
                if item["kind"] == "ndarray":
                    item.pop("dtype")
            entry_manifest = dict(
                symbol=symbol,
                definition=name + "." + definition["name"],
                arguments=argument_schema,
                captures=capture,
                dependencies=dict(methods=code.contract["methods"], functions=code.contract["functions"]),
            )
            entries[definition["name"]] = entry_manifest
            manifest["entries"][symbol] = entry_manifest
        payloads["host/" + relative.as_posix()] = _host_projection(
            name,
            tree,
            kind,
            schemas,
            definitions,
            entries,
            package.__name__,
            device_functions,
            relative.name == "__init__.py",
        ).encode("utf8")
        manifest["sources"][name] = hashlib.sha256(text.encode("utf8")).hexdigest()
    if not manifest["entries"]:
        raise PackageExportError("Package contains no supported decorated kernels")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".forge-package-", dir=output.parent))
    try:
        module.archive(str(staging / "code.tcm"))
        for relative, data in payloads.items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                manifest["files"][path.relative_to(staging).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        if any(path.read_text(encoding="utf8") != text for _, path, _, text, *_ in sources):
            raise PackageExportError("Source changed during export")
        (staging / "package.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
        staging.rename(output)
    finally:
        # This is the exact private staging directory allocated above, never user input.
        if staging.exists() and staging.parent == output.parent and staging.name.startswith(".forge-package-"):
            shutil.rmtree(staging)
    return manifest
