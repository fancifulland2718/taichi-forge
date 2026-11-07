"""Experimental definition-driven package export; no model execution tracing.

The supported slice is typed value/ndarray owners with ordinary decorators.
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
from types import ModuleType, SimpleNamespace

from taichi_forge.tools._compiled_host import decorator_resolver
from ._package_types import qualified_kind, resolve_type, type_expression, type_spec
from ._package_fields import field_declarations, construct_fields, field_allocation

_DTYPES = frozenset(("f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"))
_DECORATORS = frozenset(("kernel", "func", "real_func", "pyfunc", "data_oriented", "dataclass"))
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
    schema = field_declarations(module, constructor, kind, _fail)
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
            if target.attr in schema and schema[target.attr]["kind"] == "dense_field":
                continue
            if isinstance(value, ast.Call) and qualified_kind(value, kind) == "ndarray":
                if statement not in constructor.body or len(targets) != 1:
                    _fail(module, statement, "conditional or aliased allocation requires a resource contract")
                arguments = dict(zip(("dtype", "shape"), value.args))
                if len(value.args) > 2 or any(k.arg not in ("dtype", "shape") for k in value.keywords):
                    _fail(module, value, "only ndarray dtype and shape are supported")
                arguments.update((k.arg, k.value) for k in value.keywords)
                shape = arguments.get("shape")
                if "dtype" not in arguments or shape is None:
                    _fail(module, value, "ndarray requires a fixed element type and explicit shape")
                rank = len(shape.elts) if isinstance(shape, (ast.Tuple, ast.List)) else 1
                spec = dict(kind="ndarray", type_node=arguments["dtype"], ndim=rank)
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
    if any(
        (isinstance(n, ast.Name) and n.id.startswith(_PREFIX))
        or (isinstance(n, ast.Attribute) and n.attr.startswith(_PREFIX))
        or (isinstance(n, ast.arg) and n.arg.startswith(_PREFIX))
        or (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name.startswith(_PREFIX))
        for n in ast.walk(tree)
    ):
        _fail(module, tree, "reserved package-export identifier")
    kind = decorator_resolver(tree)
    definitions, owners = [], {}
    supported = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            decorators = [kind(d) for d in node.decorator_list]
            if "data_oriented" not in decorators and "dataclass" not in decorators:
                continue
            if decorators not in (["data_oriented"], ["dataclass"]):
                _fail(module, node, "owner decorator combinations are not yet supported")
            if decorators == ["data_oriented"]:
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
            if owner is not None and owner not in owners and dsl in ("kernel", "pyfunc"):
                _fail(
                    module, entry, "dataclass Host entries are not yet supported; device methods must be func/real_func"
                )
            if dsl not in ("kernel", "func", "real_func", "pyfunc") or decorators != [dsl]:
                _fail(module, entry, "unsupported numerical decorator combination")
            if dsl == "pyfunc":
                # Its Python body is numerical too: exporting it requires a
                # declared value domain, never a signature learned from calls.
                arguments = entry.args.args[1:] if owner is not None else entry.args.args
                if any(a.annotation is None for a in arguments):
                    _fail(module, entry, "pyfunc export requires explicit parameter types")
                if any(
                    qualified_kind(n, kind)
                    in ("lang.impl.inside_kernel", "lang.impl.inside_kernel_scope", "lang.impl.inside_python")
                    for n in ast.walk(entry)
                ):
                    _fail(module, entry, "scope-dependent pyfunc requires separate Host and numerical definitions")
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


def _host_projection(
    module, tree, kind, owners, definitions, entries, package_name, device_functions, is_package, namespace, builder
):
    numerical = {item["name"]: item for item in definitions}
    definitions_at = {(item["node"].lineno, item["node"].col_offset): item for item in definitions}
    owners_at = {
        (node.lineno, node.col_offset) for node in tree.body if isinstance(node, ast.ClassDef) and node.name in owners
    }
    imported_helpers = set()

    def internal_import(origin):
        return origin == package_name or (origin or "").startswith(package_name + ".")

    def module_import(origin):
        return ast.Call(
            ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "import_module", ast.Load()),
            [ast.Constant(origin)],
            [],
        )

    def device_import(alias, origin):
        value = namespace.get(alias.asname or alias.name)
        return f"{origin}.{alias.name}" in device_functions or (
            getattr(value, "_is_taichi_function", False) and not value.func.pyfunc
        )

    def import_origin(node):
        if node.level:
            parent = module if is_package else module.rpartition(".")[0]
            return importlib.util.resolve_name("." * node.level + (node.module or ""), parent)
        return node.module

    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            origin = import_origin(node)
            imported_helpers.update(a.asname or a.name for a in node.names if device_import(a, origin))

    class Project(ast.NodeTransformer):
        owner = None

        def visit_Assign(self, node):
            return self._allocation(node)

        def visit_AnnAssign(self, node):
            return self._allocation(node)

        def _allocation(self, node):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            schema = owners.get(self.owner, {})
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and schema.get(target.attr, {}).get("kind") == "dense_field"
                and schema[target.attr]["allocation"] == [node.lineno, node.col_offset]
            ):
                node.value = field_allocation(module + "." + self.owner, target.attr, schema)
                return node
            return self.generic_visit(node)

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
                elif internal_import(alias.name):
                    # Keep Python's import binding rules, while resolving the
                    # code owner through this loaded package's private namespace.
                    if alias.asname is None and "." in package_name:
                        _fail(module, node, "nested package absolute imports require an explicit alias")
                    value = module_import(alias.name)
                    if alias.asname is None and alias.name != package_name:
                        replacements.append(ast.Expr(value))
                        value = module_import(package_name)
                    replacements.append(
                        ast.Assign([ast.Name(alias.asname or alias.name.split(".")[0], ast.Store())], value)
                    )
                else:
                    replacements.append(ast.Import([alias]))
            return replacements

        def visit_ImportFrom(self, node):
            origin = import_origin(node)
            node.names = [a for a in node.names if not device_import(a, origin)]
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
            if not node.level and internal_import(origin):
                if any(a.name == "*" for a in node.names):
                    _fail(module, node, "wildcard imports cannot expose removed device-only definitions")
                return [
                    ast.Assign(
                        [ast.Name(a.asname or a.name, ast.Store())],
                        ast.Call(
                            ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "import_from", ast.Load()),
                            [ast.Constant(origin), ast.Constant(a.name)],
                            [],
                        ),
                    )
                    for a in node.names
                ]
            return node

        def visit_ClassDef(self, node):
            if [kind(d) for d in node.decorator_list] == ["dataclass"]:
                return ast.copy_location(
                    ast.Assign(
                        [ast.Name(node.name, ast.Store())], type_expression(type_spec(namespace[node.name], builder))
                    ),
                    node,
                )
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
            if definition["kind"] in ("func", "real_func"):
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
                and numerical.get(f"{self.owner}.{node.attr}", {}).get("kind") in ("func", "real_func")
            ):
                _fail(module, node, "Host references a device-only helper")
            role = kind(node)
            if qualified_kind(node, kind) in (
                "math.vec2",
                "math.vec3",
                "math.vec4",
                "math.mat2",
                "math.mat3",
                "math.mat4",
                "math.ivec2",
                "math.ivec3",
                "math.ivec4",
                "math.uvec2",
                "math.uvec3",
                "math.uvec4",
            ):
                return ast.copy_location(type_expression(type_spec(resolve_type(node, namespace), builder)), node)
            if role is not None and role not in _DTYPES | {"ndarray", "graph"}:
                _fail(module, node, f"Host frontend operation {role!r} needs a Runtime projection")
            return self.generic_visit(node)

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and (
                numerical.get(node.id, {}).get("kind") in ("func", "real_func") or node.id in imported_helpers
            ):
                _fail(module, node, "Host references a device-only helper")
            return node

        def visit_Call(self, node):
            if qualified_kind(node, kind) in ("types.vector", "types.matrix", "types.struct"):
                return ast.copy_location(type_expression(type_spec(resolve_type(node, namespace), builder)), node)
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


def export_package(package, output, *, arch=None):
    """Experimentally export all decorated definitions in an imported module/package.

    Selects ``arch`` on a cold Runtime or validates the currently initialized
    backend. No constructor, test, Host branch, initializer or step is executed
    to discover numerical coverage. Typed arrays/values and directly typed
    scalar owner attributes form the current supported scope.
    """
    import taichi_forge as ti
    from taichi_forge.aot._component import Array, Component, RuntimeValue
    from taichi_forge.aot.module import Module
    from taichi_forge.types.ndarray_type import NdarrayType
    from taichi_forge.types._argument_descriptor import describe_annotation
    from taichi_forge.lang.matrix import MatrixType
    from taichi_forge.lang.struct import StructType

    if arch is not None:
        from taichi_forge.lang import impl

        targets = {"cpu": ti.cpu, "x64": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}
        selected = targets.get(arch) if isinstance(arch, str) else arch
        if selected not in (ti.cpu, ti.cuda, ti.vulkan):
            raise ValueError("Package arch must be cpu, cuda or vulkan")
        if impl.get_runtime().prog is None:
            ti.init(arch=selected, enable_fallback=False)
        elif impl.current_cfg().arch != selected:
            raise ValueError(
                "Package arch differs from the live JIT Runtime; initialize the chosen backend in a fresh process"
            )

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
        module_root=package.__name__,
        discovery="all_decorated_definitions",
        entries={},
        files={},
        sources={},
        owners={},
        field_groups={},
    )
    payloads = {}
    device_functions = {
        name + "." + definition["name"]
        for name, _, _, _, _, _, _, definitions in sources
        for definition in definitions
        if definition["kind"] in ("func", "real_func")
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
            declarations = construct_fields(owner, schema, vars(imported), module)
            for attribute, item in schema.items():
                if item["kind"] == "dense_field":
                    continue
                elif item["kind"] == "ndarray":
                    dtype = resolve_type(item.pop("type_node"), vars(imported))
                    item["element"] = type_spec(dtype, module)
                    declarations[attribute] = Array(dtype, item["ndim"])
                    # Structural ABI witness only: every extent remains a runtime parameter.
                    value = ti.ndarray(dtype, (1,) * item["ndim"])
                else:
                    dtype = getattr(ti, item["dtype"])
                    declarations[attribute] = RuntimeValue(dtype)
                    value = 0.0 if item["dtype"].startswith("f") else 0
                setattr(owner, attribute, value)
            owners[class_name], environments[class_name] = owner, Component(**declarations)
            manifest["owners"][name + "." + class_name] = {
                attribute: {key: value for key, value in item.items() if key not in ("allocation", "dimension_sources")}
                for attribute, item in schema.items()
            }
            if any(item["kind"] == "dense_field" for item in schema.values()):
                from ._component import _dense_group

                layout, members, dimensions, _ = _dense_group(environments[class_name], owner, "fields")
                manifest["field_groups"][name + "." + class_name] = dict(
                    schema=layout, members=members, dimensions=sorted(dimensions)
                )
        entries = {}
        for definition in definitions:
            if definition["kind"] not in ("kernel", "pyfunc"):
                continue
            owner_name, node = definition["owner"], definition["node"]
            owner = owners.get(owner_name)
            entry = getattr(owner if owner is not None else imported, node.name)
            if definition["kind"] == "pyfunc":
                from taichi_forge.lang.kernel_impl import Kernel, AutodiffMode

                # Compile the original function definition with its declared
                # entry signature. In-kernel calls still use the normal Func;
                # no application decorator or alternate numerical body exists.
                kernel = Kernel(entry.func.func, AutodiffMode.NONE, _classkernel=owner is not None)
                entry = SimpleNamespace(_primal=kernel, _kernel_owner=owner)
            kernel = getattr(entry, "_primal", None)
            if kernel is None:
                _fail(name, node, "decorated definition did not resolve to a Forge kernel")
            arguments = kernel.arguments[1:] if owner is not None else kernel.arguments
            values, argument_schema = [], []
            for argument in arguments:
                annotation = argument.annotation
                descriptor = describe_annotation(annotation)
                if descriptor.element is None:
                    _fail(name, node, f"argument {argument.name!r} requires an explicit value or ndarray type")
                dtype = str(descriptor.element.scalar_type)
                if isinstance(annotation, NdarrayType):
                    if not descriptor.element.is_complete or type(annotation.ndim) is not int or annotation.needs_grad:
                        _fail(name, node, "ndarray arguments require explicit element type/rank and no gradients")
                    values.append(ti.ndarray(annotation.dtype, (1,) * annotation.ndim))
                    argument_schema.append(
                        dict(
                            name=argument.name,
                            kind="ndarray",
                            element=type_spec(annotation.dtype, module),
                            ndim=annotation.ndim,
                        )
                    )
                elif isinstance(annotation, (MatrixType, StructType)):
                    values.append(annotation(0) if isinstance(annotation, MatrixType) else annotation())
                    argument_schema.append(
                        dict(name=argument.name, kind="compound", element=type_spec(annotation, module))
                    )
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
                if item["kind"] in ("ndarray", "dense_field"):
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
            hasattr(imported, "__path__"),
            vars(imported),
            module,
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
