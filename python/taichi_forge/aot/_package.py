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
from ._package_fields import field_declarations, field_allocation
from ._package_sources import collect_sources
from ._package_owners import resolve_owners
from ._package_errors import is_error_interface, project_error_interface
from ._package_resources import scope_declarations
from ._package_targets import BACKEND_NAMES, backend_constant, backend_query, select_target_branches
from ._package_records import dimension_relations
from ._package_constants import resolve_constants
from ._package_construction import (
    construction_functions,
    discover_construction_methods,
    materialize_inherited_construction,
)

_DTYPES = frozenset(("f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"))
_DECORATORS = frozenset(("kernel", "func", "real_func", "pyfunc", "data_oriented", "dataclass"))
_PREFIX = "_forge_package_"
_HOST_OPS = frozenset(("ndarray", "graph", "sync"))
_HOST_RESOURCE_TYPES = frozenset(("Field", "ScalarField", "MatrixField", "StructField"))
_PRODUCER_ALLOCATIONS = frozenset(
    ("root", "FieldsBuilder", "field", "axes", "Vector.field", "Matrix.field", "Struct.field", "types.struct.field")
)


class PackageExportError(ValueError):
    """A definition cannot yet be represented by the package exporter."""


def _fail(module, node, message):
    raise PackageExportError(f"{module}:{getattr(node, 'lineno', 1)}: {message}")


def _owner_schema(module, node, kind, namespace):
    """Infer types from syntax, never run __init__ or select a model branch."""
    if node.keywords:
        _fail(module, node, "custom owner metaclasses require further support")
    schema = {}
    for function in construction_functions(node):
        for name, declaration in _construction_schema(module, node, function, kind, namespace, schema).items():
            if name in schema and schema[name]["kind"] != "value":
                previous = schema[name]
                origins = ("scope_allocation", "scope_allocations")
                if declaration["kind"] == previous["kind"] == "dimension" and (
                    {k: v for k, v in declaration.items() if k not in origins}
                    == {k: v for k, v in previous.items() if k not in origins}
                ):
                    previous["scope_allocations"] = [
                        *previous.get("scope_allocations", [previous["scope_allocation"]]),
                        declaration["scope_allocation"],
                    ]
                    continue
                _fail(module, function, f"Repeated construction resource declaration {name!r}")
            schema[name] = declaration
    return schema


def _construction_schema(module, node, constructor, kind, namespace, earlier=None):
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
    for name, declaration in scope_declarations(
        module, constructor, kind, _fail, namespace, earlier, owner_node=node
    ).items():
        if name in schema and schema[name]["kind"] != "value":
            _fail(module, constructor, f"ResourceScope role {name!r} conflicts with a resource allocation")
        schema[name] = declaration
        if "presence_parameter" in declaration:
            declaration["presence_owner"] = module + "." + node.name
    resolve_constants(module, node, schema, namespace, _fail, constructor=constructor)
    return schema


def _analyze(module, source, namespace, target):
    tree = ast.parse(source.removeprefix("\ufeff"))
    if any(
        (isinstance(n, ast.Name) and n.id.startswith(_PREFIX))
        or (isinstance(n, ast.Attribute) and n.attr.startswith(_PREFIX))
        or (isinstance(n, ast.arg) and n.arg.startswith(_PREFIX))
        or (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name.startswith(_PREFIX))
        for n in ast.walk(tree)
    ):
        _fail(module, tree, "reserved package-export identifier")
    kind = decorator_resolver(tree)
    construction = materialize_inherited_construction(tree, namespace, module, _fail, kind)
    construction.extend(discover_construction_methods(tree, namespace, module, _fail, kind))
    tree._package_target_selections = select_target_branches(tree, kind, target, module, _fail)
    from ._package_factories import normalize_factories
    from ._package_contexts import discover_contexts

    tree._package_resource_contexts = discover_contexts(tree, namespace, module, _fail)
    tree._package_factory_declarations = construction + normalize_factories(tree, namespace, module, _fail, kind)
    from ._package_allocations import normalize_allocation_helpers

    tree._package_factory_declarations.extend(normalize_allocation_helpers(tree, namespace, kind, module, _fail))
    from ._package_type_factories import normalize_type_member_factories

    tree._package_factory_declarations.extend(normalize_type_member_factories(tree, namespace, kind, module, _fail))
    definitions, owners = [], {}
    supported = set()
    from ._package_closures import discover

    tree._package_closure_factories, closures = discover(module, tree, kind, namespace, _fail)
    definitions.extend(closures)
    supported.update(id(item["node"]) for item in closures)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            decorators = [kind(d) for d in node.decorator_list]
            inherited_owner = getattr(namespace.get(node.name), "_data_oriented", False)
            device_mixin = any(
                isinstance(n, ast.FunctionDef)
                and any(kind(d) in ("kernel", "func", "real_func", "pyfunc") for d in n.decorator_list)
                for n in node.body
            )
            declared_owner = any(
                any(
                    isinstance(part, ast.Call)
                    and qualified_kind(part, kind) == "experimental.ResourceScope"
                    or getattr(part, "_package_resource_context", None) is not None
                    for part in ast.walk(n)
                )
                for n in construction_functions(node)
            )
            if (
                "data_oriented" not in decorators
                and "dataclass" not in decorators
                and not inherited_owner
                and not device_mixin
                and not declared_owner
            ):
                continue
            if decorators not in ([], ["data_oriented"], ["dataclass"]):
                _fail(module, node, "owner decorator combinations are not yet supported")
            if decorators == ["data_oriented"] or inherited_owner or declared_owner:
                owners[node.name] = _owner_schema(module, node, kind, namespace)
            dataclass_owner = decorators == ["dataclass"]
            members = [(node.name, n) for n in node.body]
        else:
            dataclass_owner = False
            members = [(None, node)]
        for owner, entry in members:
            if not isinstance(entry, ast.FunctionDef):
                continue
            decorators = [kind(d) for d in entry.decorator_list]
            dsl = next((d for d in decorators if d in _DECORATORS), None)
            if dsl is None:
                continue
            if owner is not None and dataclass_owner and dsl in ("kernel", "pyfunc"):
                _fail(
                    module, entry, "dataclass Host entries are not yet supported; device methods must be func/real_func"
                )
            if dsl not in ("kernel", "func", "real_func", "pyfunc") or decorators != [dsl]:
                _fail(module, entry, "unsupported numerical decorator combination")
            error_interface = is_error_interface(entry, namespace)
            if dsl == "pyfunc" and not error_interface:
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
                dict(
                    owner=owner,
                    node=entry,
                    kind=dsl,
                    name=".".join(p for p in (owner, entry.name) if p),
                    requires_owner=owner is not None and owner not in owners and not dataclass_owner,
                    error_interface=error_interface,
                )
            )
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(kind(d) in _DECORATORS for d in node.decorator_list)
            and id(node) not in supported
        ):
            _fail(module, node, "nested, inherited, or asynchronous numerical definitions require further support")
    return ast.fix_missing_locations(tree), kind, owners, definitions


def _host_projection(
    module,
    tree,
    kind,
    owners,
    definitions,
    entries,
    package_name,
    device_functions,
    is_package,
    namespace,
    builder,
    application_modules=None,
    error_interfaces=frozenset(),
    host_contexts=None,
):
    input_families = getattr(tree, "_package_input_families", {})
    context_schemas = owners
    owners = dict(owners)
    for base, variants in input_families.items():
        owners[base] = context_schemas[variants[0]["owner"].rpartition(".")[2]]
    numerical = {item["name"]: item for item in definitions}
    definitions_at = {
        (item["node"].lineno, item["node"].col_offset): item for item in definitions if not item.get("inherited")
    }
    owners_at = {
        (node.lineno, node.col_offset) for node in tree.body if isinstance(node, ast.ClassDef) and node.name in owners
    }
    imported_helpers = set()

    def internal_import(origin):
        if application_modules is not None:
            return origin in application_modules
        return origin == package_name or (origin or "").startswith(package_name + ".")

    def module_import(origin):
        return ast.Call(
            ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "import_module", ast.Load()),
            [ast.Constant(origin)],
            [],
        )

    def device_import(alias, origin):
        if f"{origin}.{alias.name}" in error_interfaces:
            return False
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
        minimum_version = 1

        def __init__(self):
            self.producer_only = []
            self.host_functions = []

        def producer_allocation(self, node, role):
            if not self.host_functions:
                _fail(module, node, f"Unprojected {role!r} allocation is eager Host code")
            self.minimum_version = max(self.minimum_version, 18)
            self.producer_only.append(
                dict(module=module, operation=role, line=node.lineno, function=self.host_functions[-1])
            )
            return ast.copy_location(
                ast.Call(
                    ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "producer_allocation", ast.Load()),
                    [ast.Constant(role), ast.Constant(f"{module}:{node.lineno}")],
                    [],
                ),
                node,
            )

        def visit_Assign(self, node):
            return self._allocation(node)

        def visit_AnnAssign(self, node):
            return self._allocation(node)

        def _allocation(self, node):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            schema = owners.get(self.owner, {})
            placement = next(
                (
                    item["placement"]
                    for item in schema.values()
                    if item.get("placement", {}).get("declaration") == [node.lineno, node.col_offset]
                ),
                None,
            )
            if placement is not None:
                self.minimum_version = max(self.minimum_version, 10)
                node.value = ast.Call(
                    ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "placement", ast.Load()),
                    [ast.Name("self", ast.Load()), ast.parse(placement["expression"], mode="eval").body],
                    [],
                )
                return node
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and schema.get(target.attr, {}).get("kind") == "dense_field"
                and schema[target.attr]["allocation"] == [node.lineno, node.col_offset]
            ):
                variants = input_families.get(self.owner)
                if variants:
                    value = None
                    for variant in reversed(variants):
                        identity = variant["owner"]
                        allocation = field_allocation(
                            identity, target.attr, context_schemas[identity.rpartition(".")[2]]
                        )
                        value = (
                            allocation
                            if value is None
                            else ast.IfExp(
                                ast.Call(
                                    ast.Attribute(
                                        ast.Name(_PREFIX + "runtime", ast.Load()), "has_contract", ast.Load()
                                    ),
                                    [ast.Name("self", ast.Load()), ast.Constant(identity)],
                                    [],
                                ),
                                allocation,
                                value,
                            )
                        )
                    node.value = value
                else:
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
                elif alias.name == "importlib" or (alias.name.startswith("importlib.") and alias.asname is None):
                    if alias.name != "importlib":
                        replacements.append(ast.Import([alias]))
                    replacements.append(
                        ast.Assign(
                            [ast.Name(alias.asname or "importlib", ast.Store())],
                            ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "importlib", ast.Load()),
                        )
                    )
                elif alias.name.startswith(("taichi.", "taichi_forge.")):
                    _fail(module, node, "frontend submodule imports require further Host projection support")
                elif internal_import(alias.name):
                    # Keep Python's import binding rules, while resolving the
                    # code owner through this loaded package's private namespace.
                    if alias.asname is None and "." in package_name and application_modules is None:
                        _fail(module, node, "nested package absolute imports require an explicit alias")
                    value = module_import(alias.name)
                    root = alias.name.split(".")[0] if application_modules is not None else package_name
                    if alias.asname is None and alias.name != root:
                        replacements.append(ast.Expr(value))
                        value = module_import(root)
                    replacements.append(
                        ast.Assign([ast.Name(alias.asname or alias.name.split(".")[0], ast.Store())], value)
                    )
                else:
                    replacements.append(ast.Import([alias]))
            return replacements

        def visit_ImportFrom(self, node):
            origin = import_origin(node)
            if host_contexts is not None:
                selected, remaining = [], []
                for alias in node.names:
                    value = namespace.get(alias.asname or alias.name)
                    if host_contexts.contains(value):
                        helper_module, helper_name = host_contexts.select(value)
                        selected.append(
                            ast.Assign(
                                [ast.Name(alias.asname or alias.name, ast.Store())],
                                ast.Call(
                                    ast.Attribute(
                                        ast.Name(_PREFIX + "runtime", ast.Load()), "context_function", ast.Load()
                                    ),
                                    [ast.Constant(helper_module), ast.Constant(helper_name)],
                                    [],
                                ),
                            )
                        )
                    else:
                        remaining.append(alias)
                if selected:
                    node.names = remaining
                    rest = self.visit_ImportFrom(node) if remaining else None
                    return [*selected, *(rest if isinstance(rest, list) else [rest] if rest else [])]
            node.names = [a for a in node.names if not device_import(a, origin)]
            if not node.names:
                return None
            if origin == "importlib" and not node.level:
                if any(a.name == "*" for a in node.names):
                    _fail(module, node, "wildcard importlib imports cannot preserve the package import boundary")
                ordinary = [a for a in node.names if a.name != "import_module"]
                return ([ast.ImportFrom(origin, ordinary, level=0)] if ordinary else []) + [
                    ast.Assign(
                        [ast.Name(a.asname or a.name, ast.Store())],
                        ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "dynamic_import", ast.Load()),
                    )
                    for a in node.names
                    if a.name == "import_module"
                ]
            if not node.level and node.module in ("taichi", "taichi_forge"):
                names = [a for a in node.names if a.name not in _DECORATORS]
                tensors = [a for a in names if a.name in ("Vector", "Matrix")]
                names = [a for a in names if a not in tensors]
                if tensors:
                    self.minimum_version = max(self.minimum_version, 8)
                if any(a.name not in _DTYPES | _HOST_OPS | BACKEND_NAMES for a in names):
                    _fail(module, node, "unsupported frontend dependency in Host code")
                return [
                    ast.Assign(
                        [ast.Name(a.asname or a.name, ast.Store())],
                        (
                            ast.Call(
                                ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "arch", ast.Load()),
                                [ast.Constant(backend_constant(ast.Name(a.asname or a.name, ast.Load()), kind))],
                                [],
                            )
                            if a.name in BACKEND_NAMES
                            else ast.Attribute(
                                ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "host", ast.Load()),
                                a.name,
                                ast.Load(),
                            )
                        ),
                    )
                    for a in names
                ] + [
                    ast.Assign(
                        [ast.Name(a.asname or a.name, ast.Store())],
                        ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "tensor", ast.Load()),
                    )
                    for a in tensors
                ]
            if not node.level and (node.module or "").startswith(("taichi.", "taichi_forge.")):
                if node.module in ("taichi.lang._ndarray", "taichi_forge.lang._ndarray"):
                    from ..lang._ndarray import Ndarray

                    imported = importlib.import_module(node.module)
                    if all(a.name == "Ndarray" and getattr(imported, a.name, None) is Ndarray for a in node.names):
                        self.minimum_version = max(self.minimum_version, 17)
                        return [
                            ast.Assign(
                                [ast.Name(a.asname or a.name, ast.Store())],
                                ast.Call(
                                    ast.Attribute(
                                        ast.Name(_PREFIX + "runtime", ast.Load()), "resource_type", ast.Load()
                                    ),
                                    [ast.Constant("ndarray")],
                                    [],
                                ),
                            )
                            for a in node.names
                        ]
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

        def visit_With(self, node):
            recipe = getattr(node, "_package_resource_context", None)
            if recipe is not None:
                call = node.items[0].context_expr
                group = module + "." + self.owner + f":context_{node.lineno}_{node.col_offset}"
                helper_module, helper_name = recipe["definition"].rsplit(".", 1)
                node.items[0].context_expr = ast.copy_location(
                    ast.Call(
                        ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "resource_context", ast.Load()),
                        [ast.Constant(helper_module), ast.Constant(helper_name), *call.args],
                        [*call.keywords, ast.keyword("_forge_package_group", ast.Constant(group))],
                    ),
                    call,
                )
                self.minimum_version = max(self.minimum_version, 16)
            return self.generic_visit(node)

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
                for definition in definitions:
                    if (
                        definition["owner"] == node.name
                        and definition.get("inherited")
                        and definition["kind"] in ("kernel", "pyfunc")
                        and not definition.get("error_interface")
                    ):
                        node.body.append(
                            ast.Assign(
                                [ast.Name(definition["node"].name, ast.Store())],
                                ast.Call(
                                    ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "entry", ast.Load()),
                                    [ast.Constant(entries[definition["name"]]["symbol"])],
                                    [],
                                ),
                            )
                        )
            return node

        def visit_FunctionDef(self, node):
            nonlocal kind
            definition = definitions_at.get((node.lineno, node.col_offset))
            if definition is None:
                if host_contexts is not None and self.owner is None:
                    node = host_contexts.project_definition(module, node)
                previous = kind
                local = decorator_resolver(node)
                kind = lambda expression: local(expression) or previous(expression)
                try:
                    # Defaults, annotations and decorators execute in the outer
                    # scope, unlike the deferred body of ordinary Host methods.
                    node.args = self.visit(node.args)
                    node.decorator_list = [self.visit(d) for d in node.decorator_list]
                    if node.returns is not None:
                        node.returns = self.visit(node.returns)
                    self.host_functions.append(".".join(n for n in (self.owner, node.name) if n))
                    try:
                        body = self.visit(ast.Module(body=node.body, type_ignores=[]))
                        node.body = body.body
                    finally:
                        self.host_functions.pop()
                    return node
                finally:
                    kind = previous
            if definition.get("error_interface"):
                return self.generic_visit(project_error_interface(node, module + "." + definition["name"]))
            if definition.get("closure_definition"):
                self.minimum_version = max(self.minimum_version, 25)
                options = definition.get("closure_options", {})
                if options or definition.get("conditional_factory"):
                    self.minimum_version = max(self.minimum_version, 31)
                names = definition["closure_names"]
                descriptor = ast.Call(
                    ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "closure_entry", ast.Load()),
                    [
                        ast.Constant(module + "." + definition["name"]),
                        ast.Dict(
                            [ast.Constant(name) for name in names], [ast.Name(name, ast.Load()) for name in names]
                        ),
                    ],
                    (
                        [
                            ast.keyword(
                                "options",
                                ast.Dict(
                                    [ast.Constant(key) for key in options], [copy.deepcopy(v) for v in options.values()]
                                ),
                            )
                        ]
                        if options
                        else []
                    ),
                )
                return ast.copy_location(ast.Assign([ast.Name(node.name, ast.Store())], descriptor), node)
            if definition["kind"] in ("func", "real_func"):
                return None
            if definition.get("requires_owner"):
                descriptor = ast.Call(
                    ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "unbound_entry", ast.Load()),
                    [ast.Constant(module + "." + definition["name"])],
                    [],
                )
                return ast.copy_location(ast.Assign([ast.Name(node.name, ast.Store())], descriptor), node)
            descriptor = ast.Call(
                ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "entry", ast.Load()),
                [ast.Constant(entries[definition["name"]]["symbol"])],
                [],
            )
            return ast.copy_location(ast.Assign([ast.Name(node.name, ast.Store())], descriptor), node)

        def visit_Attribute(self, node):
            role = qualified_kind(node, kind)
            if isinstance(node.ctx, ast.Load) and role in _PRODUCER_ALLOCATIONS:
                return self.producer_allocation(node, role)
            if isinstance(node.ctx, ast.Load) and role in _HOST_RESOURCE_TYPES:
                self.minimum_version = max(self.minimum_version, 18)
                return ast.copy_location(
                    ast.Call(
                        ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "resource_type", ast.Load()),
                        [ast.Constant(role)],
                        [],
                    ),
                    node,
                )
            if isinstance(node.ctx, ast.Load) and role in ("Layout.AOS", "Layout.SOA"):
                self.minimum_version = max(self.minimum_version, 18)
                return ast.copy_location(
                    ast.Call(
                        ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "layout", ast.Load()),
                        [ast.Constant(role.split(".")[-1])],
                        [],
                    ),
                    node,
                )
            if role == "lang.impl.get_runtime.prog" and isinstance(node.ctx, ast.Load):
                if not isinstance(node.value, ast.Call) or node.value.args or node.value.keywords:
                    _fail(module, node, "Runtime identity query takes no arguments")
                self.minimum_version = max(self.minimum_version, 9)
                return ast.copy_location(
                    ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "program", ast.Load()), node
                )
            if role in ("graph", "sync"):
                self.minimum_version = max(self.minimum_version, 9)
            if role in ("aot.Component", "aot.Module") and isinstance(node.ctx, ast.Load):
                self.minimum_version = max(self.minimum_version, 9)
                self.producer_only.append(dict(module=module, operation=role, line=node.lineno))
                return ast.copy_location(
                    ast.Call(
                        ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "producer_type", ast.Load()),
                        [ast.Constant("ti." + role)],
                        [],
                    ),
                    node,
                )
            if isinstance(node.ctx, ast.Load):
                if backend_query(node, kind):
                    return ast.copy_location(
                        ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "target", ast.Load()), node
                    )
                backend = backend_constant(node, kind)
                if backend is not None:
                    return ast.copy_location(
                        ast.Call(
                            ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "arch", ast.Load()),
                            [ast.Constant(backend)],
                            [],
                        ),
                        node,
                    )
            if isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name):
                cls = namespace.get(node.value.id)
                if inspect.isclass(cls):
                    value = next((vars(base)[node.attr] for base in cls.__mro__ if node.attr in vars(base)), None)
                    if getattr(value, "_is_wrapped_kernel", False) or getattr(value, "_is_taichi_function", False):
                        _fail(module, node, "unbound numerical references require a qualified receiver binding")
            if (
                isinstance(node.ctx, ast.Load)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and numerical.get(f"{self.owner}.{node.attr}", {}).get("kind") in ("func", "real_func")
                and not numerical[f"{self.owner}.{node.attr}"].get("error_interface")
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
                self.minimum_version = max(self.minimum_version, 8)
                return ast.copy_location(type_expression(type_spec(resolve_type(node, namespace), builder)), node)
            if role is not None and role not in _DTYPES | _HOST_OPS:
                _fail(module, node, f"Host frontend operation {role!r} needs a Runtime projection")
            return self.generic_visit(node)

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and (
                (
                    numerical.get(node.id, {}).get("kind") in ("func", "real_func")
                    and not numerical[node.id].get("error_interface")
                )
                or node.id in imported_helpers
            ):
                _fail(module, node, "Host references a device-only helper")
            return node

        def visit_Subscript(self, node):
            if self.host_functions and isinstance(node.slice, ast.Constant) and node.slice.value is None:
                self.minimum_version = max(self.minimum_version, 18)
            return self.generic_visit(node)

        def visit_Call(self, node):
            for item in owners.get(self.owner, {}).values():
                if item.get("reference_declaration") == [
                    getattr(node, "lineno", None),
                    getattr(node, "col_offset", None),
                ]:
                    self.minimum_version = max(self.minimum_version, 22)
                    return ast.copy_location(
                        ast.Call(
                            ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "owner_reference", ast.Load()),
                            [
                                ast.Constant(module + "." + self.owner),
                                self.visit(node.func.value),
                                *(self.visit(arg) for arg in node.args),
                            ],
                            [ast.keyword(k.arg, self.visit(k.value)) for k in node.keywords],
                        ),
                        node,
                    )
            for item in owners.get(self.owner, {}).values():
                if item.get("borrow_declaration") == [getattr(node, "lineno", None), getattr(node, "col_offset", None)]:
                    self.minimum_version = max(self.minimum_version, 20)
                    return ast.copy_location(
                        ast.Call(
                            ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "borrow", ast.Load()),
                            [
                                ast.Constant(module + "." + self.owner),
                                self.visit(node.func.value),
                                *(self.visit(arg) for arg in node.args),
                            ],
                            [ast.keyword(k.arg, self.visit(k.value)) for k in node.keywords],
                        ),
                        node,
                    )
            for item in owners.get(self.owner, {}).values():
                if item.get("group_builder", {}).get("allocation") == [
                    getattr(node, "lineno", None),
                    getattr(node, "col_offset", None),
                ]:
                    self.minimum_version = max(self.minimum_version, 10)
                    return ast.copy_location(
                        ast.Call(
                            ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "group_builder", ast.Load()),
                            [
                                ast.Name("self", ast.Load()),
                                ast.Constant(module + "." + self.owner),
                                ast.Constant(item["group"]),
                            ],
                            [],
                        ),
                        node,
                    )
            role = qualified_kind(node, kind)
            if role in ("types.vector", "types.matrix"):
                self.minimum_version = max(self.minimum_version, 9)
                node.func = ast.Attribute(
                    ast.Name(_PREFIX + "runtime", ast.Load()), role.split(".")[-1] + "_type", ast.Load()
                )
                return self.generic_visit(node)
            if role == "types.struct" and self.host_functions:
                self.minimum_version = max(self.minimum_version, 35)
                node.func = ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "struct_type", ast.Load())
                return self.generic_visit(node)
            if qualified_kind(node, kind) in ("Vector", "Matrix"):
                self.minimum_version = max(self.minimum_version, 8)
                node.func = ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "tensor", ast.Load())
                return self.generic_visit(node)
            if qualified_kind(node, kind) == "experimental.ResourceScope":
                if (
                    len(node.args) == 1
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "self"
                    and any(
                        [node.lineno, node.col_offset] in item.get("scope_allocations", [item.get("scope_allocation")])
                        for item in owners.get(self.owner, {}).values()
                    )
                ):
                    # The normal scope forwards attributes to this same owner;
                    # declarations are projected separately at their source sites.
                    return ast.copy_location(
                        ast.Call(
                            ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "resource_scope", ast.Load()),
                            [ast.Name("self", ast.Load())],
                            [],
                        ),
                        node,
                    )
                _fail(module, node, "Package ResourceScope builder/initialization requires further support")
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Name)
                and node.func.value.func.id == "super"
                and numerical.get(f"{self.owner}.{node.func.attr}", {}).get("kind") in _DECORATORS
            ):
                _fail(module, node, "explicit super numerical calls require a qualified receiver binding")
            if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                _fail(module, node, "builtin dynamic import requires an explicit importlib boundary")
            if qualified_kind(node, kind) in ("types.vector", "types.matrix", "types.struct", "types.ndarray"):
                try:
                    spec = type_spec(resolve_type(node, namespace, allow_ndarray=True), builder)
                except (KeyError, TypeError, ValueError) as error:
                    _fail(module, node, f"Host type declaration requires fixed types: {error}")
                if spec["kind"] == "ndarray_annotation":
                    self.minimum_version = max(self.minimum_version, 7)
                elif spec["kind"] == "tensor":
                    self.minimum_version = max(self.minimum_version, 8)
                return ast.copy_location(type_expression(spec), node)
            if kind(node) in _DTYPES:
                _fail(module, node, "Host scalar casts require further Runtime projection support")
            return self.generic_visit(node)

        def visit_Expr(self, node):
            aliases = getattr(node, "_package_alias_mapping", None)
            if aliases is not None:
                self.minimum_version = max(self.minimum_version, 36)
                source, expected = aliases
                return ast.copy_location(ast.Expr(ast.Call(
                    ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "check_alias_mapping", ast.Load()),
                    [self.visit(source), ast.Constant(expected)], [],
                )), node)
            constant = next(
                (
                    item
                    for item in owners.get(self.owner, {}).values()
                    if item.get("constant_declaration") == [node.lineno, node.col_offset]
                ),
                None,
            )
            if constant is not None:
                # Keep the ordinary declaration's position and attribute read.
                name = node.value.args[0].value
                self.minimum_version = max(self.minimum_version, 30)
                return ast.copy_location(
                    ast.Expr(
                        ast.Call(
                            ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "declare_constant", ast.Load()),
                            [ast.Name("self", ast.Load()), ast.Constant(name)],
                            [],
                        )
                    ),
                    node,
                )
            declaration = next(
                (
                    (name, item)
                    for name, item in owners.get(self.owner, {}).items()
                    if item.get("value_declaration") == [node.lineno, node.col_offset]
                ),
                None,
            )
            if declaration is not None:
                name, item = declaration
                return ast.copy_location(
                    ast.Expr(
                        ast.Call(
                            ast.Attribute(ast.Name(_PREFIX + "runtime", ast.Load()), "declare_value", ast.Load()),
                            [
                                ast.Name("self", ast.Load()),
                                ast.Constant(name),
                                ast.parse(repr(item["element"]), mode="eval").body,
                                ast.parse(repr(item.get("path", [name])), mode="eval").body,
                            ],
                            [],
                        )
                    ),
                    node,
                )
            return self.generic_visit(node)

    projector = Project()
    projected = projector.visit(copy.deepcopy(tree))
    factory_definitions = {fact["definition"] for fact in tree._package_factory_declarations}
    loaded_names = {n.id for n in ast.walk(projected) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}

    class DeclarationImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            retained = []
            for alias in node.names:
                local = alias.asname or alias.name
                value = namespace.get(local)
                definition = (value.__module__ + "." + value.__qualname__) if inspect.isfunction(value) else None
                if local in loaded_names or definition not in factory_definitions:
                    retained.append(alias)
            node.names = retained
            return node if retained else None

    projected = DeclarationImports().visit(projected)
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
    return ast.unparse(projected) + "\n", projector.minimum_version, projector.producer_only


def export_package(package, output, *, arch=None, caps=None, source_root=None):
    """Experimentally export all decorated definitions in an imported module/package.

    Selects ``arch`` on a cold Runtime or validates the currently initialized
    backend. No constructor, test, Host branch, initializer or step is executed
    to discover numerical coverage. Typed arrays/values and directly typed
    scalar owner attributes form the current supported scope.
    """
    import taichi_forge as ti
    from taichi_forge.aot._component import Component
    from taichi_forge_runtime._static_values import static_key
    from ._package_variants import compile_static_variants
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
    if source_root is not None and output.is_relative_to(Path(source_root).resolve()):
        raise ValueError("Package output must be outside its application source root")
    paths, namespaces, dependency_report = collect_sources(package, source_root)
    program = ti.lang.impl.get_runtime().prog
    if program is None:
        raise PackageExportError("Initialize the Runtime or select an export backend")
    target_arch = ti.lang.impl.current_cfg().arch
    application_modules = set(paths) | set(namespaces) if source_root is not None else None
    sources, failures, imported_modules = [], [], {}
    for name, (path, relative) in paths.items():
        text = path.read_text(encoding="utf8")
        imported = package if name == package.__name__ else importlib.import_module(name)
        if ti.lang.impl.get_runtime().prog is not program or ti.lang.impl.current_cfg().arch != target_arch:
            raise PackageExportError(f"Module import changed the selected export Runtime: {name}")
        if Path(imported.__file__).resolve() != path:
            raise PackageExportError(f"Module origin differs from scanned source: {name}")
        imported_modules[name] = imported
        try:
            tree, kind, owners, definitions = _analyze(name, text, vars(imported), target_arch.name)
        except PackageExportError as error:
            failures.append(str(error))
            continue
        sources.append((name, path, relative, text, tree, kind, owners, definitions))
    if failures:
        raise PackageExportError("Package declaration preflight failed:\n" + "\n".join(failures))
    from ._package_contexts import HostContexts
    from ._package_factories import _source_value

    host_contexts = HostContexts(imported_modules)
    for name, _, _, _, tree, *_ in sources:
        for node in ast.walk(tree):
            recipe = getattr(node, "_package_resource_context", None)
            if recipe is not None:
                function = _source_value(node.items[0].context_expr.func, vars(imported_modules[name]))
                host_contexts.add(function, recipe)
    resolve_owners(sources, imported_modules, _owner_schema, _fail)
    from ._package_owner_sequences import resolve_sequences

    resolve_sequences(sources, imported_modules)
    from ._package_inputs import resolve_inputs

    resolve_inputs(sources, imported_modules)
    from ._package_field_like import resolve_field_likes

    resolve_field_likes(sources)
    from ._package_closures import calls as closure_calls, materialize as materialize_closure

    factory_calls = closure_calls(sources, imported_modules, _fail)
    module = Module(caps=caps)
    from ._package_owner_graph import OwnerGraph, owner_bindings, layout_owner

    owner_graph = OwnerGraph(sources, imported_modules, module)
    manifest = dict(
        version=(
            3
            if any(
                definition.get("inherited") or definition.get("requires_owner")
                for _, _, _, _, _, _, _, definitions in sources
                for definition in definitions
            )
            else 1
        ),
        kind="forge_definition_package_prototype",
        target=module._arch.name,
        target_capabilities=list(caps or ()),
        module_root=package.__name__,
        discovery="all_decorated_definitions",
        entries={},
        entry_families={},
        closure_entries={},
        files={},
        sources={},
        owners={},
        field_groups={},
        producer_only=[],
        declaration_factories=[
            fact for _, _, _, _, tree, _, _, _ in sources for fact in tree._package_factory_declarations
        ],
        target_selections={
            name: tree._package_target_selections
            for name, _, _, _, tree, _, _, _ in sources
            if tree._package_target_selections
        },
        owner_templates=[
            name + "." + definition["name"]
            for name, _, _, _, _, _, _, definitions in sources
            for definition in definitions
            if definition.get("requires_owner")
            and not definition.get("closure_definition")
            and definition["kind"] in ("kernel", "pyfunc")
            and not definition.get("error_interface")
        ],
        input_families={
            name + "." + base: variants
            for name, _, _, _, tree, *_ in sources
            for base, variants in getattr(tree, "_package_input_families", {}).items()
        },
        error_interfaces=[
            name + "." + definition["name"]
            for name, _, _, _, _, _, _, definitions in sources
            for definition in definitions
            if definition.get("error_interface")
        ],
    )
    payloads = {}
    if manifest["input_families"]:
        manifest["version"] = 43
    if manifest["target_selections"] or any(
        backend_query(node, kind) or backend_constant(node, kind) is not None
        for _, _, _, _, tree, kind, _, _ in sources
        for node in ast.walk(tree)
    ):
        manifest["version"] = 6
    if any(
        "value_declaration" in item
        for _, _, _, _, _, _, schemas, _ in sources
        for schema in schemas.values()
        for item in schema.values()
    ):
        manifest["version"] = max(5, manifest["version"])
    if any(
        "scope_allocation" in item and item["kind"] != "value"
        for _, _, _, _, _, _, schemas, _ in sources
        for schema in schemas.values()
        for item in schema.values()
    ):
        manifest["version"] = max(5, manifest["version"])
    if source_root is not None:
        manifest.update(
            version=max(2, manifest["version"]),
            application_modules=sorted(application_modules),
            entry_module=package.__name__,
            dependencies=dependency_report,
        )
        payloads["host/__init__.py"] = b""
        # Namespace parents remain namespaces; no application initializer is
        # invented or removed. Real initializers are ordinary scanned sources.
    device_functions = {
        name + "." + definition["name"]
        for name, _, _, _, _, _, _, definitions in sources
        for definition in definitions
        if definition["kind"] in ("func", "real_func") and not definition.get("error_interface")
    }
    shared_entries, pyfunc_kernels, closure_kernels = {}, {}, {}
    for name, path, relative, text, tree, kind, schemas, definitions in sources:
        imported = imported_modules[name]
        owners = {}
        environments = {}
        compilation_schemas = {}
        for class_name in schemas:
            owner, schema, environment, expanded = owner_graph.construct(name + "." + class_name)
            schemas[class_name] = schema
            owners[class_name], environments[class_name] = owner, environment
            compilation_schemas[class_name] = expanded
            if any(item["kind"] == "owner" for item in schema.values()):
                manifest["version"] = max(21, manifest["version"])
            if any(item["kind"] == "owner_input" for item in schema.values()):
                manifest["version"] = max(22, manifest["version"])
            if any("reference_path" in item for item in schema.values()):
                manifest["version"] = max(38, manifest["version"])
            if any("like_source_path" in item for item in schema.values()):
                manifest["version"] = max(39, manifest["version"])
            if any(item.get("struct_borrow") for item in schema.values()):
                manifest["version"] = max(41, manifest["version"])
            if any(item.get("owner_dimension") for item in schema.values()):
                manifest["version"] = max(42, manifest["version"])
            if any(item["kind"] == "host_owner_sequence" for item in schema.values()):
                manifest["version"] = max(44, manifest["version"])
            if any(item["kind"] == "borrowed_field" and "input_parameter" in item for item in schema.values()):
                manifest["version"] = max(23, manifest["version"])
            if any(item.get("optional_input") for item in schema.values()):
                manifest["version"] = max(24, manifest["version"])
            if any(item["kind"] == "static" for item in schema.values()):
                manifest["version"] = max(11, manifest["version"])
            if any(item["kind"] == "constant" for item in schema.values()):
                manifest["version"] = max(30, manifest["version"])
            if any(item.get("affects_plan") for item in schema.values()):
                manifest["version"] = max(33, manifest["version"])
            if any(item.get("dimension_alias_checks") for item in schema.values()):
                manifest["version"] = max(34, manifest["version"])
            if any(item.get("local_type_source") is not None or "member_initialization" in item for item in schema.values()):
                manifest["version"] = max(35, manifest["version"])
            if any("presence_parameter" in item for item in schema.values()):
                manifest["version"] = max(13, manifest["version"])
            if any(item["kind"] == "record" or item.get("constraints") for item in schema.values()):
                manifest["version"] = max(19, manifest["version"])
            if any(item["kind"] == "dimension" and item.get("expression") for item in schema.values()):
                manifest["version"] = max(15, manifest["version"])
            if any(item.get("struct_field") for item in schema.values()):
                manifest["version"] = max(14, manifest["version"])
            if any(item.get("optional_presence") for item in schema.values()):
                manifest["version"] = max(13, manifest["version"])
            manifest["owners"][name + "." + class_name] = {
                attribute: {
                    key: value
                    for key, value in item.items()
                    if key
                    not in (
                        "allocation",
                        "dimension_sources",
                        "local_type_source",
                        "dimension_alias_checks",
                        "declaration_module",
                        "declaration_owner",
                        "value_declaration",
                        "static_declaration",
                        "constant_declaration",
                        "constant_from_class",
                        "scope_allocation",
                        "scope_allocations",
                        "group_builder",
                        "placement",
                        "placement_alias",
                        "borrow_declaration",
                        "reference_declaration",
                        "constructor_call",
                        "default_constructor_call",
                        "order_source",
                        "sequence_declaration",
                        "constructors",
                    )
                }
                for attribute, item in schema.items()
            }
            for attribute, item in schema.items():
                if item["kind"] == "host_owner_sequence":
                    manifest["owners"][name + "." + class_name][attribute]["owner_types"] = sorted(
                        {child["owner_type"] for child in item["constructors"]}
                    )
            if any(item["kind"] == "dense_field" for item in schema.values()):
                from ._component import _dense_group

                owner_key = name + "." + class_name
                for group in sorted(
                    {item.get("group", "fields") for item in schema.values() if item["kind"] == "dense_field"}
                ):
                    layout, members, dimensions, _ = _dense_group(environments[class_name], owner, group)
                    if any(paths[0] == "schema_nodes" for paths in members.values()):
                        manifest["version"] = max(26, manifest["version"])
                    key = owner_key if group == "fields" else owner_key + ":" + group
                    manifest["field_groups"][key] = dict(schema=layout, members=members, dimensions=sorted(dimensions))
                    member_schemas = {
                        attribute: schema[
                            environments[class_name]._source_scope_paths.get(
                                attribute, environments[class_name]._paths[attribute]
                            )[0]
                        ]
                        for attribute in members
                    }
                    conditions = {
                        attribute: member_schemas[attribute]["conditions"]
                        for attribute in members
                        if member_schemas[attribute].get("conditions")
                    }
                    if conditions:
                        if any(paths[0] == "schema_nodes" for paths in members.values()):
                            raise PackageExportError(
                                "Conditional model tensor groups require projected layout selection"
                            )
                        manifest["field_groups"][key]["conditions"] = conditions
                        containers = {
                            tuple(members[attribute][0][:-1]): member_schemas[attribute]["placement"]["conditions"]
                            for attribute in members
                            if "placement" in member_schemas[attribute]
                        }
                        manifest["field_groups"][key]["containers"] = [
                            dict(path=list(path), conditions=guard) for path, guard in sorted(containers.items())
                        ]
                        manifest["version"] = max(12, manifest["version"])
                    orders = {
                        attribute: member_schemas[attribute]["order_policy"]
                        for attribute in members
                        if member_schemas[attribute].get("order_policy")
                    }
                    if orders:
                        from taichi_forge_runtime.aot._package_layout import select_layout

                        contract = manifest["field_groups"][key]
                        contract["orders"] = orders
                        selection = {policy["condition"]: True for policy in orders.values()}
                        selection.update({flag: True for guards in conditions.values() for flag in guards})
                        # Validate each independent layout choice, not a Cartesian
                        # product of owner configurations or numerical entries.
                        select_layout(contract, selection)
                        for flag in {policy["condition"] for policy in orders.values()}:
                            select_layout(contract, {**selection, flag: False})
                        manifest["version"] = max(40, manifest["version"])
                    for attribute in members:
                        path = environments[class_name]._paths[attribute]
                        item = manifest["owners"][owner_key][path[0]]
                        item["field_group"] = key
                        if len(path) > 1:
                            item.setdefault("member_ids", {})[path[1]] = attribute
        entries = {}
        compiled_definitions = []
        input_families = getattr(tree, "_package_input_families", {})
        for definition in definitions:
            variants = input_families.get(definition["owner"])
            if variants:
                for variant in variants:
                    context = variant["owner"].rpartition(".")[2]
                    compiled_definitions.append(
                        {
                            **definition,
                            "owner": context,
                            "name": context + definition["name"][len(definition["owner"]) :],
                        }
                    )
            else:
                compiled_definitions.append(definition)
        for class_name in schemas:
            for attribute, call, recipe in factory_calls.get(name + "." + class_name, ()):
                closure_environment = copy.copy(environments[class_name])
                try:
                    kernel, bindings, selected = materialize_closure(
                        recipe, call, owners[class_name], closure_environment, closure_kernels
                    )
                except (KeyError, TypeError, ValueError) as error:
                    _fail(name, call, str(error))
                compiled_definitions.append(
                    dict(
                        owner=class_name,
                        node=selected["node"],
                        kind="kernel",
                        name=class_name + "." + attribute,
                        closure_kernel=kernel,
                        closure_bindings=bindings,
                        closure_identity=selected["identity"],
                        closure_options=selected["options"],
                        closure_template_owner=selected["template_owner"],
                        closure_environment=closure_environment,
                    )
                )
        for definition in compiled_definitions:
            if (
                definition["kind"] not in ("kernel", "pyfunc")
                or definition.get("requires_owner")
                or definition.get("error_interface")
            ):
                continue
            owner_name, node = definition["owner"], definition["node"]
            owner = owners.get(owner_name)
            entry = (
                definition["closure_kernel"]
                if "closure_kernel" in definition
                else getattr(owner if owner is not None else imported, node.name)
            )
            if definition.get("closure_template_owner"):
                # Reuse Component's explicit template receiver. No generated
                # numerical wrapper or per-model template object is compiled.
                entry = SimpleNamespace(_primal=entry._primal, _kernel_owner=owner)
            if definition["kind"] == "pyfunc":
                from taichi_forge.lang.kernel_impl import Kernel, AutodiffMode

                # Compile the original function definition with its declared
                # entry signature. In-kernel calls still use the normal Func;
                # no application decorator or alternate numerical body exists.
                identity = (entry.func.func, owner is not None)
                if identity not in pyfunc_kernels:
                    pyfunc_kernels[identity] = Kernel(
                        entry.func.func, AutodiffMode.NONE, _classkernel=owner is not None
                    )
                kernel = pyfunc_kernels[identity]
                entry = SimpleNamespace(_primal=kernel, _kernel_owner=owner)
            kernel = getattr(entry, "_primal", None)
            if kernel is None:
                _fail(name, node, "decorated definition did not resolve to a Forge kernel")
            arguments = (
                kernel.arguments[1:]
                if owner is not None
                and ("closure_kernel" not in definition or definition.get("closure_template_owner"))
                else kernel.arguments
            )
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
            environment = (
                definition["closure_environment"]
                if "closure_environment" in definition
                else environments[owner_name] if owner_name is not None else Component()
            )
            try:
                codes = compile_static_variants(
                    environment, entry, values, owner, compilation_schemas.get(owner_name, {})
                )
            except Exception as error:
                raise PackageExportError(f"{name}.{definition['name']}: {error}") from error
            variants = []
            for code in codes:
                for attribute in {key for key, _ in code.contract["captures"]} | code.static.keys():
                    required = compilation_schemas.get(owner_name, {}).get(attribute, {}).get("conditions", {})
                    if any(code.static.get(key) != static_key(value) for key, value in required.items()):
                        _fail(
                            name,
                            node,
                            f"Conditional resource {attribute!r} requires its declared static guard in the numerical entry",
                        )
                # Sharing is decided by the compiler object/key and its actual
                # ABI/dependency guards, never by a semantic hash alone.
                bound_used = {
                    key: value
                    for key, value in code.used.items()
                    if all(
                        code.static.get(flag) == static_key(expected)
                        for flag, expected in compilation_schemas.get(owner_name, {})
                        .get(key, {})
                        .get("conditions", {})
                        .items()
                    )
                }
                constraints, dimensions, _ = environment.constraint_facts(owner, bound_used)
                input_presence = {
                    key: dict(path=list(environment._paths[key]), present=key in bound_used)
                    for key, item in compilation_schemas.get(owner_name, {}).items()
                    if key in code.used and item.get("optional_input")
                }
                source_bindings = {
                    key: dict(path=list(environment._paths[key]), source_path=item["source_path"])
                    for key, item in compilation_schemas.get(owner_name, {}).items()
                    if key in bound_used and item["kind"] == "borrowed_field"
                }
                child_bindings = owner_bindings(environment, compilation_schemas.get(owner_name, {}), bound_used)
                constants = {
                    key: dict(path=list(environment._paths[key]), value=value)
                    for key, value in code.static.items()
                    if compilation_schemas.get(owner_name, {}).get(key, {}).get("kind") == "constant"
                }
                admission = (
                    dict(
                        constraints=constraints,
                        dimensions=dimension_relations(environment, dimensions),
                        **({"source_bindings": source_bindings} if source_bindings else {}),
                        **({"owner_bindings": child_bindings} if child_bindings else {}),
                        **({"input_presence": input_presence} if input_presence else {}),
                        **({"constants": constants} if constants else {}),
                    )
                    if constraints or source_bindings or child_bindings or input_presence or constants
                    else None
                )
                identity = (code.kernel, code.key)
                if identity in shared_entries:
                    entry_manifest = shared_entries[identity]
                    if entry_manifest.get("admission") != admission:
                        raise PackageExportError("Shared package code requires compatible dimension admission")
                    if name + "." + definition["name"] not in entry_manifest["definitions"]:
                        entry_manifest["definitions"].append(name + "." + definition["name"])
                else:
                    # Semantic capture identity alone does not describe the
                    # per-kernel compilation policy. Target and module options
                    # are common to this export; the decorator override is not.
                    compile_options = {"opt_level": code.kernel.opt_level}
                    code_identity = json.dumps([code.semantic_key, compile_options], sort_keys=True).encode("utf8")
                    symbol = "definition_" + hashlib.sha256(code_identity).hexdigest()
                    code.export(module, symbol)
                    capture = [
                        {
                            "name": capture_name,
                            **(
                                {"path": list(environment._paths[capture_name])}
                                if capture_name not in code.workspaces and len(environment._paths[capture_name]) > 1
                                else {}
                            ),
                            "kind": schema[0],
                            "dtype": schema[1],
                            **({"element_shape": list(schema[2])} if schema[0] == "value" and schema[2] else {}),
                        }
                        for capture_name, schema in code.contract["captures"]
                    ]
                    for item in capture:
                        if item["kind"] == "local_tensor_workspace":
                            item.update(code.workspaces[item["name"]])
                            manifest["version"] = max(27, manifest["version"])
                        if item["kind"] == "model_tensor_field":
                            shape = environment.declarations[item["name"]].element_shape
                            item["model_element_shape"] = list(shape)
                            item["model_dimension_relations"] = dimension_relations(
                                environment, (n for n in shape if isinstance(n, str))
                            )
                        if item["kind"] in ("dense_field", "model_tensor_field"):
                            item.update(
                                layout_owner(environment, compilation_schemas.get(owner_name, {}), item["name"])
                            )
                        if item["kind"] in ("ndarray", "dense_field", "model_tensor_field"):
                            item.pop("dtype")
                        if item["kind"] == "dimension" and environment.declarations[item["name"]].expression:
                            relations = dimension_relations(environment, (item["name"],))
                            if environment._derived_attributes.intersection(relations):
                                item["dimension_relations"] = relations
                    entry_manifest = dict(
                        symbol=symbol,
                        definition=name + "." + definition["name"],
                        definitions=[name + "." + definition["name"]],
                        arguments=argument_schema,
                        captures=capture,
                        dependencies=dict(methods=code.contract["methods"], functions=code.contract["functions"]),
                        compile_options=compile_options,
                    )
                    if code.contract.get("model_static_ranges"):
                        manifest["version"] = max(28, manifest["version"])
                        entry_manifest["model_static_ranges"] = code.contract["model_static_ranges"]
                    if code.contract.get("model_blocks"):
                        manifest["version"] = max(29, manifest["version"])
                        entry_manifest["model_blocks"] = code.contract["model_blocks"]
                    manifest["entries"][symbol] = entry_manifest
                    shared_entries[identity] = entry_manifest
                if admission:
                    entry_manifest["admission"] = admission
                variants.append(
                    dict(
                        selection={key: value for key, value in code.static.items() if key not in constants},
                        symbol=entry_manifest["symbol"],
                    )
                )
            if any(variant["selection"] for variant in variants):
                # Logical method identity is separate from its shared numerical
                # entries. Unread owner options are absent from this family.
                # One Host sequence position may select several factory
                # compilation options. Each family retains its own code map.
                family_identity = json.dumps([name + "." + definition["name"], variants], sort_keys=True).encode()
                symbol = "family_" + hashlib.sha256(family_identity).hexdigest()
                used = {key for variant in variants for key in variant["selection"]}
                manifest["entry_families"][symbol] = dict(
                    arguments=argument_schema,
                    variants=variants,
                    domain={key: compilation_schemas[owner_name][key]["choices"] for key in sorted(used)},
                    **(
                        {"paths": {key: list(environment._paths[key]) for key in used}}
                        if any(len(environment._paths[key]) > 1 for key in used)
                        else {}
                    ),
                )
                conditions = {
                    key: compilation_schemas[owner_name][key]["conditions"]
                    for key in sorted(used)
                    if compilation_schemas[owner_name][key].get("conditions")
                }
                if conditions:
                    manifest["version"] = max(manifest["version"], 37)
                    manifest["entry_families"][symbol]["conditions"] = conditions
                entries[definition["name"]] = dict(symbol=symbol)
            else:
                entries[definition["name"]] = entry_manifest
            if "closure_kernel" in definition:
                manifest["version"] = max(manifest["version"], 25)
                if definition["closure_template_owner"]:
                    manifest["version"] = max(manifest["version"], 32)
                manifest["closure_entries"].setdefault(definition["closure_identity"], []).append(
                    dict(
                        symbol=entries[definition["name"]]["symbol"],
                        owner_type=name + "." + owner_name,
                        bindings=definition["closure_bindings"],
                        **({"template_owner": True} if definition["closure_template_owner"] else {}),
                        **(
                            {
                                "options": {
                                    key: static_key(value) for key, value in definition["closure_options"].items()
                                }
                            }
                            if definition["closure_options"]
                            else {}
                        ),
                    )
                )
        for definition in definitions:
            variants = input_families.get(definition["owner"])
            if (
                not variants
                or definition["kind"] not in ("kernel", "pyfunc")
                or definition.get("requires_owner")
                or definition.get("error_interface")
            ):
                continue
            methods = {
                variant["owner"]: entries[
                    variant["owner"].rpartition(".")[2] + definition["name"][len(definition["owner"]) :]
                ]["symbol"]
                for variant in variants
            }
            identity = json.dumps([name + "." + definition["name"], methods], sort_keys=True).encode()
            symbol = "inputs_" + hashlib.sha256(identity).hexdigest()
            first = next(iter(methods.values()))
            contract = manifest["entry_families"].get(first) or manifest["entries"][first]
            manifest["entry_families"][symbol] = dict(arguments=contract["arguments"], owner_inputs=methods)
            entries[definition["name"]] = dict(symbol=symbol)
        host_source, minimum_version, producer_only = _host_projection(
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
            application_modules,
            frozenset(manifest["error_interfaces"]),
            host_contexts,
        )
        payloads["host/" + relative.as_posix()] = host_source.encode("utf8")
        manifest["version"] = max(manifest["version"], minimum_version)
        manifest["producer_only"].extend(producer_only)
        manifest["sources"][name] = hashlib.sha256(text.encode("utf8")).hexdigest()
    context_payloads, manifest["host_contexts"] = host_contexts.render()
    payloads.update(context_payloads)
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
