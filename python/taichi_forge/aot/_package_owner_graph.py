"""Resolve declared construction types without executing application constructors."""

import ast
import builtins
import copy
from types import MappingProxyType, ModuleType, SimpleNamespace


def _member_paths(node, receiver, forwarded=None, method_reads=None, nodes=None):
    """Read explicit device member paths, without traversing Host construction."""
    paths, presence = set(), set()
    nodes = tuple(ast.walk(node)) if nodes is None else nodes
    parents = ({id(child): part for part in nodes for child in ast.iter_child_nodes(part)}
               if forwarded is not None else {})
    attributes = {id(part.value) for part in nodes if isinstance(part, ast.Attribute)}
    presence_reads = {
        id(part.left) for part in nodes
        if isinstance(part, ast.Compare) and len(part.ops) == 1
        and isinstance(part.ops[0], (ast.Is, ast.IsNot))
        and isinstance(part.comparators[0], ast.Constant) and part.comparators[0].value is None
    }
    for part in nodes:
        if isinstance(part, ast.Name) and part.id == receiver and id(part) not in attributes:
            if forwarded is not None:
                resolved = forwarded(part, parents)
                if resolved is not None:
                    paths.update(resolved)
                    continue
            # An owner passed to an unresolved helper/alias can expose any of
            # its methods. Keep the existing complete device view in that case.
            paths.add(())
        if not isinstance(part, ast.Attribute) or id(part) in attributes:
            continue
        if method_reads is not None:
            resolved = method_reads(part, parents)
            if resolved is not None:
                paths.update(resolved)
                continue
        value = part
        path, presence_only = [], id(part) in presence_reads
        while isinstance(part, ast.Attribute):
            path.append(part.attr)
            part = part.value
        if isinstance(part, ast.Name) and part.id == receiver:
            prefix = tuple(reversed(path))
            if forwarded is not None and not presence_only:
                resolved = forwarded(value, parents)
                if resolved is not None:
                    # Passing self.child to an exact helper has the same
                    # dependency proof as passing a named owner parameter.
                    # Keep its actual member reads, not a whole-child escape
                    # that recursively restores unrelated Host backreferences.
                    paths.update((*prefix, *member) for member in resolved)
                    continue
            (presence if presence_only else paths).add(prefix)
    # A nullable check does not expose all methods of a child whose concrete
    # query is already named in this body. A bare owner alias/argument still does.
    paths.update(path for path in presence
                 if not any(len(member) > len(path) and member[:len(path)] == path for member in paths))
    return paths


class _MethodInputPaths:
    """Summarize exact declared helper calls, then discard the syntax index."""

    def __init__(self, methods, modules=None, functions=None, static_function=None):
        self.methods, self.cache = methods, {}
        self.modules, self.functions = modules or {}, functions or {}
        self.static_function = static_function

    def parameter(self, owner, method, parameter, visiting=frozenset(), literals=()):
        return self._parameter(owner, self.methods[owner][method], parameter, visiting, literals, True)

    def _parameter(self, owner, node, parameter, visiting, literals, is_method):
        key = owner, id(node), parameter, tuple((name, type(value), value) for name, value in literals)
        recursion_key = owner, id(node), parameter
        if recursion_key in visiting:
            return frozenset(((),))
        if key in self.cache:
            return self.cache[key]
        parameters = (*node.args.posonlyargs, *node.args.args)
        receiver = parameters[0].arg if is_method else None
        values, unknown = dict(literals), object()
        local_names = {arg.arg for arg in (*parameters, *node.args.kwonlyargs)}
        rebound = {part.id for part in ast.walk(node)
                   if isinstance(part, ast.Name) and isinstance(part.ctx, ast.Store)}
        local_names.update(rebound)
        for name in rebound:
            values.pop(name, None)
        module = self.modules.get(id(node))
        namespace = vars(module) if module is not None else {}
        # Rebound receivers and nested lexical scopes do not prove the callee.
        known_receiver = not any(
            isinstance(part, ast.Name) and part.id in (receiver, parameter) and isinstance(part.ctx, ast.Store)
            or isinstance(part, (ast.FunctionDef, ast.Lambda)) and part is not node
            for part in ast.walk(node))

        def literal(part):
            if isinstance(part, ast.Constant) and isinstance(part.value, (str, int, float, bool, type(None))):
                return part.value
            if isinstance(part, ast.Name):
                return values.get(part.id, unknown)
            if isinstance(part, ast.BinOp) and isinstance(part.op, ast.Add):
                left, right = literal(part.left), literal(part.right)
                if isinstance(left, str) and isinstance(right, str):
                    return left + right
            if isinstance(part, ast.Compare) and len(part.ops) == 1:
                left, right = literal(part.left), literal(part.comparators[0])
                if (left is not unknown and right is not unknown
                        and (left is None or left is True or left is False
                             or right is None or right is True or right is False)):
                    if isinstance(part.ops[0], ast.Is):
                        return left is right
                    if isinstance(part.ops[0], ast.IsNot):
                        return left is not right
            if isinstance(part, ast.BoolOp):
                for value in part.values:
                    result = literal(value)
                    if result is unknown:
                        return unknown
                    if isinstance(part.op, ast.And) and not result or isinstance(part.op, ast.Or) and result:
                        return result
                return result
            if isinstance(part, ast.UnaryOp) and isinstance(part.op, ast.Not):
                value = literal(part.operand)
                return unknown if value is unknown else not value
            return unknown

        def bound_symbol(symbol):
            suffix = []
            while isinstance(symbol, ast.Attribute):
                suffix.append(symbol.attr)
                symbol = symbol.value
            if not isinstance(symbol, ast.Name) or symbol.id in local_names:
                return None
            target = namespace.get(symbol.id)
            for name in reversed(suffix):
                if not isinstance(target, ModuleType):
                    return None
                target = vars(target).get(name)
            return target

        def active_nodes():
            pending, result = [node], []
            while pending:
                part = pending.pop()
                result.append(part)
                if (isinstance(part, ast.If) and isinstance(part.test, ast.Call)
                        and self.static_function is not None
                        and bound_symbol(part.test.func) is self.static_function
                        and len(part.test.args) == 1 and not part.test.keywords):
                    value = literal(part.test.args[0])
                    if value is not unknown:
                        pending.extend(reversed(part.body if value else part.orelse))
                        continue
                pending.extend(reversed(tuple(ast.iter_child_nodes(part))))
            return result

        def declared_call(call):
            if (not known_receiver or any(isinstance(arg, ast.Starred) for arg in call.args)
                    or any(arg.arg is None for arg in call.keywords)):
                return None
            if (is_method and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == receiver
                    and call.func.attr in self.methods[owner]):
                callee, bound = self.methods[owner][call.func.attr], True
                if any(getattr(decorator, 'id', getattr(decorator, 'attr', '')) in ('staticmethod', 'classmethod')
                       for decorator in (*node.decorator_list, *callee.decorator_list)):
                    return None
            else:
                callee, bound = self.functions.get(id(bound_symbol(call.func))), False
                if callee is None:
                    return None
            all_positional = (*callee.args.posonlyargs, *callee.args.args)
            positional = all_positional[1:] if bound else all_positional
            if len(call.args) > len(positional):
                return None
            arguments = {arg.arg: value for arg, value in zip(positional, call.args)}
            allowed = {arg.arg for arg in (*positional, *callee.args.kwonlyargs)}
            for arg in call.keywords:
                if arg.arg not in allowed or arg.arg in arguments:
                    return None
                arguments[arg.arg] = arg.value
            constants = {}
            for arg, default in zip(all_positional[-len(callee.args.defaults):], callee.args.defaults):
                if arg.arg not in arguments and isinstance(default, ast.Constant):
                    constants[arg.arg] = default.value
            for arg, default in zip(callee.args.kwonlyargs, callee.args.kw_defaults):
                if arg.arg not in arguments and isinstance(default, ast.Constant):
                    constants[arg.arg] = default.value
            constants.update((name, value) for name, expr in arguments.items()
                             if (value := literal(expr)) is not unknown)
            return callee, bound, arguments, tuple(sorted(constants.items()))

        def forwarded(value, parents):
            parent = parents.get(id(value))
            call = parents.get(id(parent)) if isinstance(parent, ast.keyword) else parent
            if not known_receiver or not isinstance(call, ast.Call):
                return None
            if (isinstance(call.func, ast.Name) and call.func.id == 'getattr'
                    and 'getattr' not in local_names and namespace.get('getattr', builtins.getattr) is builtins.getattr
                    and len(call.args) in (2, 3) and value is call.args[0]):
                member = literal(call.args[1])
                return {(member,)} if isinstance(member, str) else None
            resolved = declared_call(call)
            if resolved is None:
                return None
            callee, bound, arguments, constants = resolved
            target = next((name for name, argument in arguments.items() if argument is value), None)
            return (self._parameter(owner, callee, target, visiting | {recursion_key}, constants, bound)
                    if target is not None else None)

        def method_reads(part, parents):
            call = parents.get(id(part))
            if (parameter != receiver or not isinstance(call, ast.Call) or call.func is not part
                    or not isinstance(part.value, ast.Name) or part.value.id != receiver):
                return None
            resolved = declared_call(call)
            if resolved is None:
                return None
            callee, bound, _, constants = resolved
            if not bound:
                return None
            callee_receiver = (*callee.args.posonlyargs, *callee.args.args)[0].arg
            return self._parameter(owner, callee, callee_receiver, visiting | {recursion_key}, constants, True)

        result = frozenset(_member_paths(node, parameter, forwarded, method_reads, active_nodes()))
        if parameter in rebound or any(
                isinstance(part, ast.Call) and isinstance(part.func, ast.Name) and part.func.id == 'super'
                for part in ast.walk(node)):
            result = result | {()}
        self.cache[key] = result
        return result


def owner_bindings(environment, schema, used):
    paths = [environment._paths[key] for key in used if key in environment._paths]
    paths.extend(environment._source_scope_paths[key] for key in used if key in environment._source_scope_paths)
    return {
        key: dict(
            path=item.get("path", [key]),
            owner_type=item["owner_type"],
            **({"optional": True} if item.get("dynamic_input_record") else {}),
            **({"reference_path": item["reference_path"]} if "reference_path" in item else {}),
        )
        for key, item in schema.items()
        if item["kind"] in ("owner", "owner_input")
        and any(tuple(path[: len(item.get("path", [key]))]) == tuple(item.get("path", [key])) for path in paths)
    }


def layout_owner(environment, schema, name):
    from ._component import _capture_id

    path = environment._source_scope_paths.get(name, environment._paths[name])
    owners = [
        tuple(item.get("path", [key])) for key, item in schema.items() if item["kind"] in ("owner", "owner_input")
    ]
    owner = max((p for p in owners if tuple(path[: len(p)]) == p), key=len, default=())
    return dict(layout_owner_path=list(owner), layout_member=_capture_id(path[len(owner) :])) if owner else {}


class OwnerGraph:
    def __init__(self, sources, imported, module, *, field_trees):
        from ._package_input_families import source_class
        from taichi_forge.lang.impl import static

        self.imported, self.module = imported, module
        self.field_trees = field_trees
        self._contexts = {}
        self._schemas = {}
        self._resource_contexts = {}
        self._allocation_schemas = {}
        self._field_witnesses = {}
        self._call_contexts = set()
        self._call_input_paths = {}
        self._input_field_types = {}
        self.definitions = {
            name + "." + owner: (name, source_class(tree, owner), copy.deepcopy(schema))
            for name, _, _, _, tree, _, schemas, _ in sources
            for owner, schema in schemas.items()
        }
        device_methods, device_functions, node_modules = {}, {}, {}
        for name, _, _, _, tree, _, _, definitions in sources:
            for definition in definitions:
                if definition.get('error_interface'):
                    continue
                node = definition['node']
                node_modules[id(node)] = imported[name]
                if definition['owner'] is None:
                    function = vars(imported[name]).get(node.name)
                    original = getattr(getattr(function, 'func', None), 'func', None)
                    if (getattr(function, '_is_taichi_function', False)
                            and getattr(original, '__module__', None) == name
                            and getattr(original, '__qualname__', None) == node.name):
                        device_functions[id(function)] = node
                    continue
                parameters = (*node.args.posonlyargs, *node.args.args)
                if parameters:
                    owner = name + '.' + source_class(tree, definition['owner'])
                    device_methods.setdefault(owner, {})[node.name] = node
        self._device_paths, self._device_methods = {}, {}
        self._numerical_path_cache = {}
        methods = {}
        for identity, (name, cls_name, _) in self.definitions.items():
            cls = getattr(imported[name], cls_name)
            # Follow the same method override order as the original owner.
            inherited = methods[identity] = {}
            for base in reversed(cls.__mro__):
                # An opaque/Host override must not inherit the base helper's
                # proof merely because their method names match.
                for method in base.__dict__:
                    inherited.pop(method, None)
                inherited.update(device_methods.get(base.__module__ + '.' + base.__qualname__, {}))
        # These summaries survive the scan; the temporary method/parent AST
        # indices do not. They do not own or construct application resources.
        reads = _MethodInputPaths(methods, node_modules, device_functions, static)
        self._method_input_paths = {}
        for identity, nodes in methods.items():
            summaries = self._device_methods.setdefault(identity, {})
            for method, node in nodes.items():
                parameters = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
                for parameter in parameters:
                    self._method_input_paths[identity, id(node), parameter.arg] = reads.parameter(
                        identity, method, parameter.arg)
                paths = set(reads.parameter(identity, method, parameters[0].arg))
                if any(isinstance(part, ast.Call) and isinstance(part.func, ast.Name)
                       and part.func.id == 'super' for part in ast.walk(node)):
                    paths.add(())
                summaries[method] = frozenset(paths)
            self._device_paths[identity] = frozenset(path for paths in summaries.values() for path in paths)
        for identity in self.definitions:
            self._validate(identity, ())

    def _numerical_paths(self, identity, paths):
        """Expand the methods actually queried, using the discovered definitions.

        A borrowed query must not inherit the owner's unrelated producer
        kernels. Unknown receiver escapes retain the previous complete view.
        Only path tuples are cached; no extra AST or application owner is kept.
        """
        key = identity, frozenset(paths)
        if key in self._numerical_path_cache:
            return self._numerical_path_cache[key]
        pending, visited, result = list(key[1]), set(), set()
        methods = self._device_methods.get(identity, {})
        while pending:
            path = pending.pop()
            if path in visited:
                continue
            visited.add(path)
            if not path:
                result.update(self._device_paths.get(identity, ()))
                # A whole-owner escape also allows passing one of its children
                # to another helper without an explicit member access here.
                result.update(tuple(item.get('path', (name,)))
                              for name, item in self.definitions[identity][2].items()
                              if item['kind'] in ('owner', 'owner_input'))
            elif len(path) == 1 and path[0] in methods:
                pending.extend(methods[path[0]])
            else:
                result.add(path)
        result = frozenset(result)
        self._numerical_path_cache[key] = result
        return result

    def input_field_type(self, identity, path):
        """Use the declared field ABI without constructing a consumer owner."""
        key = identity, tuple(path)
        if key in self._input_field_types:
            return self._input_field_types[key]
        # Reuse allocation slicing before copying declarations. A depth input
        # must not construct its provider's unrelated children or buffers merely
        # to discover a scalar type. The existing witness pool owns the slice.
        _, _, _, schema = self.construct(identity, resource_only=True, resource_paths=(key[1],))
        for attribute, item in schema.items():
            if tuple(item.get("path", (attribute,))) != key[1]:
                continue
            if (item["kind"] != "dense_field" or item.get("dtype_policy") or item.get("shape_policy")
                    or item.get("element", {}).get("kind") not in ("scalar", "tensor")
                    or item.get("element", {}).get("model_tensor")):
                raise ValueError("Independent Field template requires a fixed element type and rank")
            # Allocation conditions govern the provider's storage, not this
            # fixed input ABI. The call binder still requires a present, typed
            # resource; inspecting its declaration does not make it optional.
            result = dict(element=item["element"], ndim=len(item["shape"]))
            self._input_field_types[key] = result
            return result
        raise ValueError("Independent Field template has no declared field path")

    def register_call_context(self, identity, inputs, fields, node):
        """Compose declared input views without an application owner or constructor."""
        schema = {"_arg_" + key: dict(kind="owner_input", owner_type=value) for key, value in inputs.items()}
        schema.update({"_arg_" + key: dict(kind="ndarray", **value) for key, value in fields.items()})
        paths = self._call_input_paths.setdefault(identity, {})
        parameters = (*node.args.posonlyargs, *node.args.args)
        receiver_type = inputs.get(parameters[0].arg) if parameters else None
        for key in inputs:
            read_paths = self._method_input_paths.get((receiver_type, id(node), key))
            paths.setdefault('_arg_' + key, set()).update(
                _member_paths(node, key) if read_paths is None else read_paths)
        if identity in self.definitions:
            if self.definitions[identity][2] != schema:
                raise ValueError("Call context changed its declared input types")
            return
        self.definitions[identity] = (*identity.rsplit(".", 1), schema)
        self._call_contexts.add(identity)

    def context(self, identity, choices=None):
        """Reuse definition witnesses across kernels, never consumer instances."""
        key = identity, tuple(sorted((choices or {}).items()))
        if key not in self._contexts:
            self._contexts[key] = self.construct(identity, choices=choices)
        if not choices:
            self._schemas[identity] = self._contexts[key][1]
        return self._contexts[key]

    def contract(self, identity):
        """Extract this owner's native layout without its unused solver graph.

        Keep the pointer-free declaration afterwards. Physical allocation groups
        still use the shared witness pool, including their source dependencies.
        """
        result = self.construct(identity, contract_only=True)
        self._schemas[identity] = result[1]
        return result

    def schema(self, identity):
        if identity not in self._schemas:
            self.contract(identity)
        return self._schemas[identity]

    def release_module(self, module):
        # Numerical owner instances are needed while this module's entries and
        # variants are captured. Later Host queries need only their schemas;
        # borrowed allocation views and native groups retain independent reuse.
        self._contexts = {
            key: value for key, value in self._contexts.items()
            if self.definitions[key[0]][0] != module
        }

    def _validate(self, identity, ancestors):
        if identity not in self.definitions:
            raise ValueError("Declared child owner is outside the supported definition closure: " + identity)
        if identity in ancestors:
            raise ValueError("Declared owner construction is cyclic: " + " -> ".join((*ancestors, identity)))
        for item in self.definitions[identity][2].values():
            if item["kind"] == "owner":
                for choice in item.get("owner_alternatives", [item]):
                    self._validate(choice["owner_type"], (*ancestors, identity))

    def construct(self, identity, *, resource_only=False, references=(), choices=None, path=(), resource_paths=None,
                  contract_only=False, numerical_paths=None):
        import taichi_forge as ti
        from taichi_forge.lang.matrix import MatrixType
        from taichi_forge_runtime._static_values import read_static_key
        from ._component import (
            Array,
            Record,
            RuntimeValue,
            StaticValue,
            _ChoiceRecord,
            _ConditionalArray,
            _LayoutArray,
            _ConditionalOwnerRecord,
            _InputPresence,
            _OptionalArray,
        )
        from ._package_fields import construct_fields, field_environment
        from ._package_records import write_member
        from ._package_types import resolve_type, type_spec, producer_static
        from ._package_borrows import project as project_borrows

        name, class_name, original = self.definitions[identity]
        schema = original
        source_children = {}
        whole_query = numerical_paths is not None and () in numerical_paths
        if numerical_paths is not None:
            numerical_paths = self._numerical_paths(identity, numerical_paths)
        query_slice = None
        if resource_only and numerical_paths is not None and not contract_only and not whole_query:
            from ._package_resource_slice import AllocationSchema

            if identity not in self._allocation_schemas:
                self._allocation_schemas[identity] = AllocationSchema(original)
            query_slice = self._allocation_schemas[identity].select_query(numerical_paths, resource_paths or ())
        if query_slice is not None:
            schema, source_children = query_slice
        elif any(item["kind"] in ("owner", "owner_input") for item in original.values()):
            from ._package_resource_slice import AllocationSchema

            # Every local layout, including a numerical owner's workspace,
            # needs its declared source dimensions and borrowed allocations.
            # Merge those dependencies with device reads before slicing input
            # owners; a dimension used only for allocation is still required.
            paths = [tuple(item.get("path", (key,))) for key, item in original.items()
                     if item["kind"] not in ("owner", "owner_input", "host_owner_sequence")]
            if identity not in self._allocation_schemas:
                self._allocation_schemas[identity] = AllocationSchema(original)
            _, source_children = self._allocation_schemas[identity].select(paths)
        if resource_paths is not None and query_slice is None:
            from ._package_resource_slice import AllocationSchema

            if identity not in self._allocation_schemas:
                self._allocation_schemas[identity] = AllocationSchema(original)
            allocation, required_children = self._allocation_schemas[identity].select(resource_paths)
            if numerical_paths is None:
                schema, source_children = allocation, required_children
            else:
                # A declined numerical slice retains its complete local view;
                # an additional layout borrow must not replace those reads.
                for child, members in required_children.items():
                    source_children.setdefault(child, set()).update(members)
        resource_key = None
        if resource_only:
            token = identity, None if resource_paths is None else tuple(sorted(resource_paths))
            if token in references:
                raise ValueError(
                    "Cyclic input resource definitions require a resolved allocation path: "
                    + identity
                    + " at "
                    + ".".join(path)
                )
            references = (*references, token)
            if resource_paths is None or numerical_paths is not None:
                # A storage borrow still omits its provider's owned graph.
                # A numerical object input also needs children explicitly read
                # by its Kernel or device methods. Keep those declared namespaces
                # before copying, without restoring unrelated Host-owned children.
                schema = {
                    key: item for key, item in schema.items()
                    if item['kind'] not in ('owner', 'owner_input')
                    or key in source_children
                    or numerical_paths is not None and any(
                        member[:len(item.get('path', (key,)))] == tuple(item.get('path', (key,)))
                        for member in numerical_paths)
                    or numerical_paths is None and item['kind'] == 'owner_input'
                }
            if len(references) == 1:
                # Reuse complete input views and complete allocation groups
                # selected by field inputs across compatible owner contexts.
                # Keep input slots distinct: two unrelated inputs in a single
                # owner must never acquire a false allocation alias. Relative
                # choices retain every layout/child selection of this source.
                # Nested borrows still traverse their cycle guards normally.
                resource_key = (
                    identity,
                    path,
                    None if numerical_paths is None else tuple(sorted(numerical_paths)),
                    tuple(schema),
                    tuple(sorted((child, tuple(sorted(paths))) for child, paths in source_children.items())),
                    tuple(
                        sorted(
                            (key[len(path) :], value)
                            for key, value in (choices or {}).items()
                            if key[: len(path)] == path
                        )
                    ),
                )
                if resource_key in self._resource_contexts:
                    return self._resource_contexts[resource_key]
        # The slice is read-only. Copy its retained declarations before any
        # per-witness selection, rather than cloning the entire source owner
        # once for each borrowed allocation.
        schema = copy.deepcopy(schema)
        static_values = {
            key: (choices or {}).get((*path, *item.get("path", (key,))), read_static_key(item["choices"][0]))
            for key, item in schema.items()
            if item["kind"] == "static"
        }

        def layout_static(policy):
            if "condition_path" not in policy:
                return static_values[policy["condition"]]
            remaining = tuple(policy["condition_path"])
            source = original
            while True:
                member = next(
                    (item for key, item in source.items() if tuple(item.get("path", (key,))) == remaining), None
                )
                if member is not None:
                    if member["kind"] != "static":
                        raise ValueError("field_like layout selection requires a declared static source")
                    return (choices or {}).get(
                        (*path, *policy["condition_path"]), read_static_key(member["choices"][0])
                    )
                child = next(
                    (
                        (tuple(item.get("path", (key,))), item)
                        for key, item in source.items()
                        if item["kind"] in ("owner", "owner_input")
                        and remaining[: len(item.get("path", (key,)))] == tuple(item.get("path", (key,)))
                    ),
                    None,
                )
                if child is None:
                    raise ValueError("field_like layout selection has no declared source path")
                prefix, item = child
                remaining = remaining[len(prefix) :]
                source = self.definitions[item["owner_type"]][2]

        for attribute, item in schema.items():
            if "source_policy" in item:
                policy = item["source_policy"]
                # Validate both complete allocation paths after child schemas
                # have been composed, including StructField members.
                selected = static_values[policy["condition"]]
                item["source_path"] = policy["if_true" if selected else "if_false"]
                item["source_guards"] = {policy["condition"]: selected}
            if "dtype_policy" in item:
                policy = item["dtype_policy"]
                selected = layout_static(policy)
                item["type_node"] = policy["if_true" if selected else "if_false"]
                item["dtype_guards"] = {policy["condition"]: selected}
            if "shape_policy" in item:
                policy = item["shape_policy"]
                selected = layout_static(policy)
                shape = policy["if_true" if selected else "if_false"]
                if item.get("external"):
                    item["shape_expressions"] = shape
                    item["ndim"] = len(shape)
                else:
                    item["shape"] = shape
                item["shape_guards"] = {policy["condition"]: selected}
            alternatives = item.get("owner_alternatives")
            if alternatives:
                applicable = [
                    option
                    for option in alternatives
                    if all(
                        static_values.get(key) is value
                        for key, value in option.get("conditions", {}).items()
                        if key not in item.get("conditions", {})
                    )
                ]
                if len(applicable) != 1:
                    raise ValueError("Child implementation choices must cover their declared guard domain")
                selected = applicable[0]
                item["owner_type"] = selected["owner_type"]
                # Constructor identity proofs belong to the selected call site,
                # just like its concrete child type. Do not lose Child(self)
                # when that child is selected by a normal Host branch.
                item["input_references"] = selected.get("input_references", {})
                item["choice_guards"] = {
                    key: value
                    for key, value in selected.get("conditions", {}).items()
                    if key not in item.get("conditions", {})
                }
        imported = self.imported[name]
        owner = SimpleNamespace() if identity in self._call_contexts else object.__new__(getattr(imported, class_name))
        # Dimensions and field_like may read child allocation metadata before
        # this owner's fields are allocated. Build each declared child once;
        # attach owned children after the parent's record namespaces exist.
        children, child_values = {}, {}
        for attribute, item in schema.items():
            if item["kind"] in ("owner", "owner_input"):
                if contract_only and attribute not in source_children:
                    continue
                child_choices = dict(choices or {})
                child_schema = self.definitions[item["owner_type"]][2]
                inputs = [
                    (tuple(child_schema[member].get("path", (member,))), target,
                     child_schema[member]["owner_type"])
                    for member, target in item.get("input_references", {}).items()
                ]
                if item["kind"] == "owner_input" and "reference_path" in item:
                    inputs.append(((), item["reference_path"], item["owner_type"]))
                for member, target, source_type in inputs:
                    source_path = (*path, *target)
                    input_path = (*path, attribute, *member)
                    # A borrowed view of the same instance must materialize
                    # the same layout before numerical variant exploration.
                    # Include default static values as well as explicit choices.
                    for key, source in self.definitions[source_type][2].items():
                        if source["kind"] == "static":
                            child_choices[(*input_path, key)] = (choices or {}).get(
                                (*source_path, key), read_static_key(source["choices"][0]))
                    for selected_path, value in (choices or {}).items():
                        if selected_path[:len(source_path)] == source_path:
                            child_choices[(*input_path, *selected_path[len(source_path):])] = value
                child_paths = source_children.get(attribute) or (
                    (tuple(schema[item["field_input"]]["source_member"]),) if "field_input" in item else None
                )
                if child_paths is None and item["kind"] == "owner_input":
                    prefix = tuple(item.get("path", (attribute,)))
                    borrowed = []
                    for declaration in schema.values():
                        if declaration["kind"] != "borrowed_field":
                            continue
                        policy = declaration.get("source_policy")
                        candidates = (
                            (policy["if_true"], policy["if_false"]) if policy else (declaration["source_path"],)
                        )
                        borrowed.extend(
                            tuple(candidate[len(prefix) :])
                            for candidate in candidates
                            if tuple(candidate[: len(prefix)]) == prefix and len(candidate) > len(prefix)
                        )
                    if any(child_schema.get(p[0], {}).get("kind") == "owner" for p in borrowed):
                        # Keep the existing input view, and include only the
                        # owned allocations explicitly borrowed through it.
                        # Do not construct the input's unrelated solver graph.
                        child_paths = tuple(
                            dict.fromkeys(
                                [
                                    *(
                                        tuple(part.get("path", (key,)))
                                        for key, part in child_schema.items()
                                        if part["kind"] != "owner"
                                    ),
                                    *borrowed,
                                ]
                            )
                        )
                child_numerical_paths = None
                if identity in self._call_contexts:
                    child_numerical_paths = self._call_input_paths[identity].get(attribute)
                elif numerical_paths is not None:
                    prefix = tuple(item.get('path', (attribute,)))
                    child_numerical_paths = tuple(member[len(prefix):] for member in numerical_paths
                                                  if member[:len(prefix)] == prefix)
                elif item["kind"] == "owner_input" and not contract_only:
                    # Ordinary owner methods use the same query dependency
                    # projection as free-kernel call contexts. A borrowed
                    # query may itself own numeric children used by its methods.
                    prefix = tuple(item.get('path', (attribute,)))
                    parent_paths = self._numerical_paths(identity, self._device_paths.get(identity, ()))
                    child_numerical_paths = tuple(member[len(prefix):] for member in parent_paths
                                                  if member[:len(prefix)] == prefix)
                value, _, child, expanded = self.construct(
                    item["owner_type"],
                    resource_only=contract_only or resource_only or item["kind"] == "owner_input",
                    references=references,
                    choices=child_choices,
                    path=(*path, attribute),
                    resource_paths=child_paths,
                    numerical_paths=child_numerical_paths,
                )
                children[attribute] = (child, expanded)
                child_values[attribute] = value
                if item["kind"] == "owner_input":
                    write_member(owner, attribute, value, schema)

        def source_dimension(parts):
            # Owned child fields may supply a clone's dimensions before they are
            # attached to the parent. Reuse that same construction witness.
            if parts[0] in child_values:
                value, parts = child_values[parts[0]], parts[1:]
            else:
                value = owner
            for part in parts:
                value = getattr(value, part)
            return value

        declarations = construct_fields(owner, schema, vars(imported), self.module,
                                        imported_modules=self.imported, source_dimension=source_dimension,
                                        field_trees=self.field_trees, witnesses=self._field_witnesses,
                                        witness_scope=(name, class_name, path))
        for attribute, item in schema.items():
            if item["kind"] in ("dense_field", "dimension", "borrowed_field", "host_owner_sequence"):
                continue
            if item["kind"] == "record":
                declarations[attribute] = Record(optional=item.get("optional", False))
                continue
            if item["kind"] in ("owner", "owner_input"):
                if attribute not in children:
                    continue
                record = (
                    _ConditionalOwnerRecord(item["conditions"].items())
                    if item.get("conditions") else Record()
                )
                if item["kind"] == "owner_input":
                    declarations[attribute] = record
                    continue
                value = child_values[attribute]
                declarations[attribute] = (
                    _ChoiceRecord(item["choice_guards"].items(), optional=bool(item.get("conditions")),
                                  conditions=item.get("conditions", {}).items())
                    if item.get("owner_alternatives")
                    else record
                )
            elif item["kind"] == "static":
                domain = tuple(producer_static(read_static_key(key)) for key in item["choices"])
                declarations[attribute] = StaticValue(choices=domain, kernel_only=item["kernel_only"])
                value = producer_static(static_values[attribute])
            elif item["kind"] == "constant":
                declarations[attribute] = StaticValue()
                value = read_static_key(item["value"])
            elif item["kind"] == "ndarray":
                namespace = vars(self.imported[item.get("declaration_module", name)])
                if "dtype_policy" in item:
                    from ._package_types import resolve_scalar_dtype_policy

                    item["dtype_policy"] = resolve_scalar_dtype_policy(item["dtype_policy"], namespace, self.module)
                if "type_node" in item:
                    dtype = resolve_type(item.pop("type_node"), namespace)
                else:
                    element = item["element"]
                    dtype = getattr(ti, element["dtype"])
                    shape = element.get("element_shape", ())
                    if shape:
                        dtype = ti.types.vector(shape[0], dtype) if len(shape) == 1 else ti.types.matrix(*shape, dtype)
                item["element"] = type_spec(dtype, self.module)
                if item.get("external") and item["element"]["kind"] not in ("scalar", "tensor"):
                    raise ValueError("Typed external borrow requires scalar/vector/matrix elements")
                conditions = {**item.get("conditions", {}), **item.get("dtype_guards", {})}
                layout_conditions = {**item.get("conditions", {}),
                                     **item.get("dtype_guards", {}), **item.get("shape_guards", {})}
                declarations[attribute] = (
                    _OptionalArray(dtype, item["ndim"], presence=item["input_presence_attribute"],
                                   conditions=tuple(layout_conditions.items()))
                    if item.get("external") and item.get("optional_input")
                    else _LayoutArray(dtype, item["ndim"], conditions=tuple(layout_conditions.items()))
                    if item.get("external") and layout_conditions
                    else (
                        _ConditionalArray(dtype, item["ndim"], conditions=tuple(conditions.items()))
                        if conditions
                        else Array(dtype, item["ndim"])
                    )
                )
                value = ti.ndarray(dtype, (1,) * item["ndim"])
            else:
                if "type_node" in item:
                    dtype = resolve_type(
                        item.pop("type_node"), vars(self.imported[item.get("declaration_module", name)])
                    )
                    item["element"] = type_spec(dtype, self.module)
                    if item["element"]["kind"] not in ("scalar", "tensor"):
                        raise ValueError("ResourceScope.value requires a scalar/vector/matrix type")
                    item["dtype"] = item["element"]["dtype"]
                else:
                    dtype = getattr(ti, item["dtype"])
                value_type = _InputPresence if item.get("input_presence") else RuntimeValue
                declarations[attribute] = value_type(dtype, affects_plan=item.get("affects_plan", False))
                value = dtype(0) if isinstance(dtype, MatrixType) else (0.0 if item["dtype"].startswith("f") else 0)
            write_member(owner, attribute, value, schema)
        from ._package_optional import project_input_records, project_boolean_groups

        project_input_records(owner, schema, declarations)
        project_boolean_groups(schema, declarations)
        environment, expanded = _compose(field_environment(declarations, schema), schema, children)
        from ._component import _capture_id

        for attribute, item in schema.items():
            if attribute not in children:
                continue
            for member, target in item.get("input_references", {}).items():
                child_schema = children[attribute][1]
                source = child_schema[member]
                key = _capture_id((attribute, *source.get("path", (member,))))
                expanded[key] = {**expanded[key], "reference_path": list(target)}
        from ._package_field_like import project_field_likes

        project_borrows(owner, environment, expanded)
        project_field_likes(environment, expanded)
        from ._package_optional import project_owned_records, project_struct_member_inputs

        project_owned_records(environment, expanded)
        project_struct_member_inputs(environment, expanded)
        result = owner, schema, environment, expanded
        if resource_key is not None:
            self._resource_contexts[resource_key] = result
        return result


def _compose(environment, schema, children):
    """Qualify existing Component declarations; native layouts remain authoritative."""
    from dataclasses import replace
    from ._component import (
        Array,
        DenseField,
        Dimension,
        Record,
        _ConditionalRecord,
        _ChoiceRecord,
        _InputRecord,
        _ModelTensorField,
        _capture_id,
    )

    declarations, paths = dict(environment.declarations), dict(environment._paths)
    constraints, source_paths = dict(environment._constraints), dict(environment._source_scope_paths)
    derived, expanded = set(environment._derived_attributes), dict(schema)
    for attribute, (child, child_schema) in children.items():
        names = {key: _capture_id((attribute, *path)) for key, path in child._paths.items()}
        # Presence controls whether the namespace exists. Implementation choice
        # controls the logical owner binding, not every field's numerical ABI.
        # Each alternative still constructs and validates its original layout.
        owner_conditions = schema[attribute].get("conditions", {})

        def expression(value):
            op, *args = value
            if op == "dim":
                return op, names[args[0]]
            return value if op == "const" else (op, *(expression(arg) for arg in args))

        for key, original in child.declarations.items():
            value = original
            if isinstance(value, Dimension):
                value = Dimension(
                    value.kind,
                    names[value.capacity] if value.capacity else "",
                    expression(value.expression) if value.expression else (),
                )
            elif isinstance(value, DenseField):
                shape = tuple(names[n] if isinstance(n, str) else n for n in value.shape)
                group = attribute + "." + value.group
                changes = dict(shape=shape, group=group)
                if isinstance(value, _ModelTensorField):
                    changes["element_shape"] = tuple(names[n] if isinstance(n, str) else n for n in value.element_shape)
                if getattr(value, "conditions", ()):
                    changes["conditions"] = tuple((names[n], expected) for n, expected in value.conditions)
                value = replace(value, **changes)
            elif isinstance(value, Array):
                changes = dict(
                    shape=(
                        None
                        if value.shape is None
                        else tuple(names[n] if isinstance(n, str) else n for n in value.shape)
                    ),
                    resource_id=names[value.resource_id] if value.resource_id else "",
                )
                if getattr(value, "conditions", ()):
                    changes["conditions"] = tuple((names[n], expected) for n, expected in value.conditions)
                if getattr(value, "presence", ""):
                    changes["presence"] = names[value.presence]
                value = replace(value, **changes)
            elif isinstance(value, _ConditionalRecord):
                value = type(value)(tuple((names[n], expected) for n, expected in value.conditions))
            elif isinstance(value, _InputRecord):
                value = _InputRecord(names[value.presence])
            elif isinstance(value, _ChoiceRecord):
                value = _ChoiceRecord(
                    tuple((names[n], expected) for n, expected in value.guards), optional=value.optional,
                    conditions=tuple((names[n], expected) for n, expected in value.conditions),
                )
            elif isinstance(value, Record):
                value = Record(optional=value.optional)
            declarations[names[key]], paths[names[key]] = value, (attribute, *child._paths[key])
        for key, binding in child._source_scope_paths.items():
            source_paths[names[key]] = (attribute, *binding)
        for key, constraint in child._constraints.items():
            spec = constraint["spec"]
            constraints[_capture_id((attribute, key))] = dict(
                source=constraint["source"],
                spec=dict(left=expression(spec["left"]), right=expression(spec["right"]), when=names[spec["when"]]),
            )
        derived.update(names[key] for key in child._derived_attributes)
        for key, item in child_schema.items():
            if item["kind"] == "host_owner_sequence":
                # Independent algorithm instances stay in their child's Host
                # contract. The collection has no device allocation/path to
                # compose into its parent; its entries are exported separately.
                continue
            transformed = {**item, "path": [attribute, *item.get("path", (key,))]}
            if "source_path" in item:
                transformed["source_path"] = [attribute, *item["source_path"]]
            if "source_policy" in item:
                policy = item["source_policy"]
                transformed["source_policy"] = {
                    "condition": names[policy["condition"]],
                    "if_true": [attribute, *policy["if_true"]],
                    "if_false": [attribute, *policy["if_false"]],
                    **(
                        {
                            "presence": {
                                branch: [
                                    {**predicate, "path": [attribute, *predicate["path"]]} for predicate in predicates
                                ]
                                for branch, predicates in policy["presence"].items()
                            }
                        }
                        if "presence" in policy
                        else {}
                    ),
                }
                transformed["source_guards"] = {names[n]: expected for n, expected in item["source_guards"].items()}
            if "input_presence_attribute" in item:
                transformed["input_presence_attribute"] = names[item["input_presence_attribute"]]
            if "input_record" in item:
                transformed["input_record"] = names[item["input_record"]]
            if "input_conditions" in item:
                transformed["input_conditions"] = [names[key] for key in item["input_conditions"]]
            if item.get("external"):
                transformed["shape_expressions"] = [expression(axis) for axis in item["shape_expressions"]]
            if "reference_path" in item:
                transformed["reference_path"] = [attribute, *item["reference_path"]]
            if "like_source_path" in item:
                transformed["like_source_path"] = [attribute, *item["like_source_path"]]
            if "source_presence" in item:
                transformed["source_presence"] = [
                    {**predicate, "path": [attribute, *predicate["path"]]} for predicate in item["source_presence"]
                ]
            if "conditions" in item:
                transformed["conditions"] = {names[n]: expected for n, expected in item["conditions"].items()}
            if "choice_guards" in item:
                transformed["choice_guards"] = {names[n]: expected for n, expected in item["choice_guards"].items()}
            if "shape_policy" in item:
                transformed["shape_policy"] = {
                    **item["shape_policy"],
                    "condition": names[item["shape_policy"]["condition"]],
                    **(
                        {"condition_path": [attribute, *item["shape_policy"]["condition_path"]]}
                        if "condition_path" in item["shape_policy"]
                        else {}
                    ),
                }
                transformed["shape_guards"] = {names[n]: expected for n, expected in item["shape_guards"].items()}
                if item.get("external"):
                    for branch in ("if_true", "if_false"):
                        transformed["shape_policy"][branch] = [expression(axis) for axis in item["shape_policy"][branch]]
            if "dtype_policy" in item:
                transformed["dtype_policy"] = {
                    **item["dtype_policy"],
                    "condition": names[item["dtype_policy"]["condition"]],
                    **(
                        {"condition_path": [attribute, *item["dtype_policy"]["condition_path"]]}
                        if "condition_path" in item["dtype_policy"]
                        else {}
                    ),
                }
                transformed["dtype_guards"] = {names[n]: expected for n, expected in item["dtype_guards"].items()}
            if "owner_alternatives" in item:
                transformed["owner_alternatives"] = [
                    {
                        **choice,
                        "conditions": {names[n]: expected for n, expected in choice.get("conditions", {}).items()},
                    }
                    for choice in item["owner_alternatives"]
                ]
            if owner_conditions:
                transformed["conditions"] = {**transformed.get("conditions", {}), **owner_conditions}
            expanded[names[key]] = transformed
    environment._set_declarations(declarations)
    environment._paths = MappingProxyType(paths)
    environment._constraints = MappingProxyType(constraints)
    environment._source_scope_paths = MappingProxyType(source_paths)
    environment._derived_attributes = frozenset(derived)
    return environment, expanded
