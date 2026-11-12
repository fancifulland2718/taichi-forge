"""Classify package configuration before visiting its finite code domain.

Only producer-side syntax and declaration facts are inspected here. Consumers
receive scalar contracts and compiled control flow, never these syntax trees.
Unproven structural uses keep the ordinary finite specialization contract.
"""

import ast
import copy

from taichi_forge.lang.ast._uniform_schedule import (
    helper_call_name, scalar_statement, uniform_range_schedule, uniform_range_statements,
)


class StaticPreparation:
    def __init__(self, sources):
        self.uses = {}
        self._contexts = {}
        self._entry_uses = {}
        self._functions = {}
        self._names = {}
        self._syntax = {}
        self._local_values = {}
        self._numeric_locals = {}
        self._value_uses = {}
        self._ambiguous_names = set()
        self._task_context_cache = {}
        self._task_context_methods = set()
        functions = []
        for module, _, _, _, tree, kind, _, _ in sources:
            tree_parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
            for function in ast.walk(tree):
                # Name-based closure analysis must not mistake a shadowing
                # parameter, callable alias or ordinary Host method for a
                # decorated helper. Ambiguity retains producer specialization.
                if isinstance(function, ast.arg):
                    self._ambiguous_names.add(function.arg)
                elif isinstance(function, ast.alias):
                    # Imported DSL helpers retain their existing resolution.
                    # Only establish whether the builtin used below is known.
                    if (function.asname or function.name.split('.')[0]) in ('getattr', '*'):
                        self._ambiguous_names.add('getattr')
                elif isinstance(function, ast.ClassDef) and function.name == 'getattr':
                    self._ambiguous_names.add('getattr')
                elif isinstance(function, (ast.Name, ast.Attribute)) and isinstance(function.ctx, ast.Store):
                    self._ambiguous_names.add(getattr(function, 'id', getattr(function, 'attr', '')))
                if not isinstance(function, ast.FunctionDef):
                    continue
                if not any(
                    kind(d) in ("kernel", "func", "real_func", "pyfunc") for d in function.decorator_list
                ):
                    self._ambiguous_names.add(function.name)
                    continue
                names, ancestor = [function.name], function
                while (ancestor := tree_parents.get(id(ancestor))) is not None:
                    if isinstance(ancestor, ast.ClassDef):
                        names.insert(0, ancestor.name)
                identity = module + "." + ".".join(names)
                if len(names) > 1:
                    self._task_context_methods.add(identity)
                functions.append((identity, function, kind))
                self._names.setdefault(function.name, set()).add(identity)
        for identity, function, kind in functions:
            parents = {id(child): node for node in ast.walk(function) for child in ast.iter_child_nodes(node)}
            self._syntax[identity] = function, parents, kind
            reads, stores, aliases = {}, {}, {}
            for node in ast.walk(function):
                if isinstance(node, ast.Name):
                    (reads if isinstance(node.ctx, ast.Load) else stores).setdefault(node.id, []).append(node)
            # Only immutable, dominating aliases are followed. Branch-local
            # assignments and rebinding retain the finite producer contract.
            for statement in function.body:
                if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                        and isinstance(statement.targets[0], ast.Name)
                        and isinstance(statement.value, ast.Call) and kind(statement.value.func) == 'static'):
                    name = statement.targets[0].id
                    if (len(stores.get(name, ())) == 1
                            and all(read.lineno > statement.end_lineno for read in reads.get(name, ()))):
                        aliases[id(statement)] = name
            if any(isinstance(node, (ast.FunctionDef, ast.Lambda)) and node is not function
                   for node in ast.walk(function)):
                aliases.clear()
            self._local_values[id(function)] = reads, stores, aliases
            self._numeric_locals[id(function)] = self._numeric_assignments(function, parents, kind, stores)
        # A call name qualifies only when every definition with that name in
        # the source closure is a proven inline schedule. Recursive/unresolved
        # groups never bootstrap themselves into this set.
        self.scalar_functions = set()
        while True:
            proven = {identity for identity, function, kind in functions
                      if any(kind(d) in ('func', 'real_func', 'pyfunc') for d in function.decorator_list)
                      and all(scalar_statement(statement, kind, self.scalar_functions, returns=True)
                              for statement in function.body)}
            names = {name for name, identities in self._names.items()
                     if name not in self._ambiguous_names and identities <= proven}
            if names <= self.scalar_functions:
                break
            self.scalar_functions.update(names)
        self.schedule_functions = set()
        while True:
            proven = {
                identity for identity, function, kind in functions
                if any(kind(d) in ("func", "pyfunc") for d in function.decorator_list)
                and uniform_range_statements(function.body, kind, self.schedule_functions, self.scalar_functions)
            }
            names = {name for name, identities in self._names.items()
                     if name not in self._ambiguous_names and identities <= proven}
            if names <= self.schedule_functions:
                break
            self.schedule_functions.update(names)
        # Inline helpers at kernel scope can contribute parallel tasks. Follow
        # their ordinary call names conservatively before moving a condition;
        # the absence of loop_config at the call site is not a legality proof.
        self.execution_functions = set()
        self.task_context_functions = set()
        calls = {}
        for _, function, kind in functions:
            if any(kind(d) == "real_func" for d in function.decorator_list):
                continue
            called = calls.setdefault(function.name, set())
            for node in ast.walk(function):
                task_context = (
                    isinstance(node, ast.Attribute) and node.attr in ("loop_config", "SharedArray", "simt")
                )
                if task_context:
                    self.task_context_functions.add(function.name)
                if isinstance(node, (ast.For, ast.While)) or task_context:
                    self.execution_functions.add(function.name)
                if isinstance(node, ast.Call):
                    called.add(getattr(node.func, "attr", getattr(node.func, "id", "")))
        while True:
            extra = {name for name, called in calls.items() if called & self.execution_functions}
            if extra <= self.execution_functions:
                break
            self.execution_functions.update(extra)
        # Reuse the production closure's conservative call graph. A helper
        # containing a barrier/shared allocation through another helper still
        # belongs to its offloaded task. Serial loops alone do not impose this
        # restriction on per-thread function units.
        while True:
            extra = {name for name, called in calls.items() if called & self.task_context_functions}
            if extra <= self.task_context_functions:
                break
            self.task_context_functions.update(extra)
        for identity, function, kind in functions:
            parents = {id(child): node for node in ast.walk(function) for child in ast.iter_child_nodes(node)}
            uses = {False: {}, True: {}}
            edges = set()
            for node in ast.walk(function):
                if isinstance(node, ast.Call):
                    callee = getattr(node.func, "attr", getattr(node.func, "id", ""))
                    edges.add((callee, self._threaded(node, parents, kind)))
                    if (
                        isinstance(node.func, ast.Name)
                        and callee in ("getattr", "hasattr")
                        and len(node.args) >= 2
                        and isinstance(node.args[1], ast.Constant)
                        and isinstance(node.args[1].value, str)
                    ):
                        # Component lowering already resolves builtin getattr
                        # through the declared owner/record attribute path. A
                        # literal lookup with an inert default has the same
                        # numerical use as self.flag. Unknown reflection and
                        # shadowed builtins must not acquire that proof.
                        declared_read = (
                            callee == 'getattr' and callee not in self._ambiguous_names
                            and len(node.args) in (2, 3) and not node.keywords
                            and isinstance(node.args[0], (ast.Name, ast.Attribute))
                            and (len(node.args) == 2 or isinstance(node.args[2], ast.Constant))
                        )
                        for threaded, context in uses.items():
                            reasons = self._use(node, parents, kind, threaded) if declared_read else {'reflective_use'}
                            context.setdefault(node.args[1].value, set()).update(reasons)
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                    for threaded, context in uses.items():
                        context.setdefault(node.attr, set()).update(self._use(node, parents, kind, threaded))
            self._functions[identity] = uses, edges
            for name, reasons in uses[False].items():
                self.uses.setdefault(name, set()).update(reasons)

    @staticmethod
    def _threaded(node, parents, kind):
        while (node := parents.get(id(node))) is not None:
            if isinstance(node, ast.For) and not (isinstance(node.iter, ast.Call) and kind(node.iter.func) == "static"):
                return True
            if isinstance(node, ast.FunctionDef) and any(kind(d) == "real_func" for d in node.decorator_list):
                return True
        return False

    def entry_uses(self, identity):
        if identity in self._entry_uses:
            return self._entry_uses[identity]
        if identity not in self._functions:
            return self.uses
        pending, visited, result = [(identity, False)], set(), {}
        while pending:
            key, threaded = pending.pop()
            if (key, threaded) in visited:
                continue
            visited.add((key, threaded))
            contexts, edges = self._functions[key]
            for name, reasons in contexts[threaded].items():
                result.setdefault(name, set()).update(reasons)
            for callee, local in edges:
                pending.extend((target, threaded or local) for target in self._names.get(callee, ()))
        self._entry_uses[identity] = result
        return result

    def requires_task_context(self, source, args):
        """Keep task-local operations inline unless template facts exclude them.

        This refines only the existing conservative call graph. No owner
        attributes, runtime values or arbitrary Python expressions are evaluated.
        """
        from taichi_forge.types.annotations import template

        if source.func.__name__ not in self.task_context_functions:
            return False
        identity = source.func.__module__ + '.' + source.func.__qualname__
        bindings = {arg.name: value for arg, value in zip(source.arguments, args)
                    if isinstance(arg.annotation, template) and type(value) is bool}
        return self._function_task_context(identity, bindings, set())

    def _function_task_context(self, identity, bindings, active):
        if identity not in self._syntax:
            return True
        key = identity, tuple(sorted(bindings.items()))
        if key in active:
            return True
        if key in self._task_context_cache:
            return self._task_context_cache[key]
        function, _, kind = self._syntax[identity]
        # Rebinding or nested scopes can invalidate a parameter's lexical fact.
        if any(isinstance(n, (ast.FunctionDef, ast.Lambda)) and n is not function for n in ast.walk(function)):
            return True
        assigned = {n.id for n in ast.walk(function) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        bindings = {name: value for name, value in bindings.items() if name not in assigned}

        def literal(node):
            if isinstance(node, ast.Constant) and type(node.value) is bool:
                return node.value
            if isinstance(node, ast.Name):
                return bindings.get(node.id)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                value = literal(node.operand)
                return None if value is None else not value
            return None

        def call_context(node, target):
            callee, _, resolve = self._syntax[target]
            parameters = [*callee.args.posonlyargs, *callee.args.args]
            if callee.args.vararg or callee.args.kwarg or callee.args.kwonlyargs:
                return True
            if any(isinstance(arg, ast.Starred) for arg in node.args) or any(k.arg is None for k in node.keywords):
                return True
            names = [arg.arg for arg in parameters]
            positional = list(node.args)
            if isinstance(node.func, ast.Attribute) and target in self._task_context_methods:
                if not names or names[0] not in ('self', 'cls'):
                    return True
                positional.insert(0, node.func.value)
            if len(positional) > len(names) or any(k.arg not in names for k in node.keywords):
                return True
            # Defaults belong to the callee's definition scope, not the caller's
            # template bindings. Only literal booleans have a scope-free value.
            defaults = [n if isinstance(n, ast.Constant) and type(n.value) is bool
                        else ast.Constant(value=None) for n in callee.args.defaults]
            values = dict(zip(names[len(names)-len(defaults):], defaults))
            values.update(zip(names, positional))
            values.update((k.arg, k.value) for k in node.keywords)
            fixed = {}
            for parameter in parameters:
                annotation = parameter.annotation
                if (isinstance(annotation, ast.Call) and resolve(annotation.func) == 'template'
                        and parameter.arg in values):
                    value = literal(values[parameter.arg])
                    if value is not None:
                        fixed[parameter.arg] = value
            return self._function_task_context(target, fixed, active | {key})

        def visit(node):
            if isinstance(node, ast.Attribute) and node.attr in ('loop_config', 'SharedArray', 'simt'):
                return True
            if isinstance(node, ast.If):
                test = node.test
                if (isinstance(test, ast.Call) and kind(test.func) == 'static'
                        and len(test.args) == 1 and not test.keywords):
                    value = literal(test.args[0])
                    if value is not None:
                        return any(visit(child) for child in (node.body if value else node.orelse))
            if isinstance(node, ast.Call):
                name = getattr(node.func, 'attr', getattr(node.func, 'id', ''))
                if name in self.task_context_functions:
                    targets = self._names.get(name, ())
                    if (name in self._ambiguous_names or not targets
                            or any(call_context(node, target) for target in targets)):
                        return True
            return any(visit(child) for child in ast.iter_child_nodes(node))

        result = any(visit(statement) for statement in function.body)
        self._task_context_cache[key] = result
        return result

    @staticmethod
    def _numeric_assignments(function, parents, kind, stores):
        """Locals already materialized as DSL values, never Python aliases."""
        parameters = {arg.arg for arg in (*function.args.posonlyargs, *function.args.args,
                                         *function.args.kwonlyargs)}
        if function.args.vararg:
            parameters.add(function.args.vararg.arg)
        if function.args.kwarg:
            parameters.add(function.args.kwarg.arg)
        if any(isinstance(part, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal))
               or isinstance(part, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef))
               and part is not function for part in ast.walk(function)):
            return {}
        result = {}
        for statement in function.body:
            if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)):
                continue
            name, value = statement.targets[0].id, statement.value
            numeric = isinstance(value, ast.Constant) and type(value.value) in (bool, int, float)
            if isinstance(value, ast.Call) and not value.keywords:
                shape_count = ({'Vector': 1, 'Matrix': 2}.get(kind(value.func.value))
                               if isinstance(value.func, ast.Attribute) and value.func.attr == 'zero' else None)
                root = value.func
                while isinstance(root, ast.Attribute):
                    root = root.value
                if (shape_count is not None and len(value.args) == shape_count + 1
                        and isinstance(root, ast.Name) and root.id not in parameters | stores.keys()
                        and kind(value.args[0]) in ('i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64',
                                                    'f16', 'f32', 'f64')
                        and all(isinstance(arg, ast.Constant) and type(arg.value) is int and arg.value > 0
                                for arg in value.args[1:])):
                    numeric = True
            if not numeric or name in parameters:
                continue
            # Normal later assignments call the existing value's _assign; a
            # static assignment instead rebinds the Python symbol. Unknown
            # binding forms (loop targets, imports, deletion, annotations) do
            # not establish the same value identity.
            valid = True
            for store in stores.get(name, ()):
                parent = parents.get(id(store))
                if ((store.lineno, store.col_offset) < (statement.lineno, statement.col_offset)
                        or not isinstance(parent, (ast.Assign, ast.AugAssign))
                        or isinstance(parent, ast.Assign)
                        and isinstance(parent.value, ast.Call) and kind(parent.value.func) == 'static'):
                    valid = False
                    break
            if valid:
                result[name] = statement.lineno
        return result

    def _structural(self, node, kind, parents):
        # These operations define types, Python objects, or execution structure.
        # Their branch alternatives cannot be represented by a scalar IR phi.
        fixed_cast_types = set()
        for part in ast.walk(node):
            if isinstance(part, ast.Call) and kind(part.func) == 'cast':
                # A literal dtype operand of the normal numeric cast creates a
                # value, not a Python type/layout alternative. Keep conditional
                # dtype expressions and every other use under the usual proof.
                dtype = part.args[1] if len(part.args) == 2 else next(
                    (kw.value for kw in part.keywords if kw.arg == 'dtype'), None)
                if isinstance(dtype, (ast.Name, ast.Attribute)) and kind(dtype) in (
                        'i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64', 'f16', 'f32', 'f64'):
                    fixed_cast_types.add(id(dtype))
            # ast.walk visits the call before its dtype operand.
            if isinstance(part, (ast.Return, ast.Raise)):
                return True
            if isinstance(part, (ast.Break, ast.Continue)):
                # The ordinary DSL lowering already handles device loop
                # transfers under a runtime predicate. Check the actual
                # lexical target, not just whether some enclosing loop is
                # parallel: an inner static loop still requires producer
                # expansion and cannot accept a runtime break/continue.
                target = parents.get(id(part))
                while target is not None and not isinstance(target, (ast.For, ast.While, ast.FunctionDef)):
                    target = parents.get(id(target))
                if not isinstance(target, (ast.For, ast.While)) or (
                    isinstance(target, ast.For) and isinstance(target.iter, ast.Call)
                    and kind(target.iter.func) == 'static'
                ):
                    return True
            if isinstance(part, ast.Attribute) and (
                part.attr in ("loop_config", "SharedArray", "Vector", "Matrix", "field", "ndarray")
                or (kind(part) or "").startswith("types.")
                or (kind(part) in ("i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f16", "f32", "f64")
                    and id(part) not in fixed_cast_types)
            ):
                return True
            if isinstance(part, ast.Assign) and any(isinstance(t, ast.Name) for t in part.targets):
                # Local Python aliases can select functions/resources/types.
                # Copying into an established DSL local is a value assignment,
                # however, and retains the ordinary _assign type/shape rules.
                if isinstance(part.value, (ast.Name, ast.Attribute, ast.Tuple, ast.List, ast.Dict)):
                    function = part
                    while function is not None and not isinstance(function, ast.FunctionDef):
                        function = parents.get(id(function))
                    numeric = self._numeric_locals.get(id(function), {})
                    if not (isinstance(part.value, ast.Name)
                            and all(isinstance(t, ast.Name) and numeric.get(t.id, part.lineno) < part.lineno
                                    for t in part.targets)):
                        return True
        return False

    def _local_use(self, function, name, parents, kind, threaded, visiting):
        key = id(function), name, threaded
        if key in visiting:
            return {'recursive_value_use'}
        if key in self._value_uses:
            return self._value_uses[key]
        reads, stores, aliases = self._local_values[id(function)]
        if name in stores and name not in aliases.values():
            return {'rebound_value'}
        reasons = set()
        for read in reads.get(name, ()):
            reasons.update(self._use(read, parents, kind, threaded, visiting | {key}))
        result = reasons or {'unused_value'}
        # Recursive failures depend on the traversal stack, so never memoize
        # them as facts for an unrelated entry.
        if 'recursive_value_use' not in result:
            self._value_uses[key] = result
        return result

    def _argument_use(self, call, child, parents, kind, threaded, visiting):
        name = helper_call_name(call)
        targets = self._names.get(name, ()) if name not in self._ambiguous_names else ()
        if not targets or any(isinstance(arg, ast.Starred) for arg in call.args):
            return {'unresolved_argument_use'}
        keyword = child.arg if isinstance(child, ast.keyword) else None
        if keyword is None and child not in call.args:
            return {'non_condition_use'}
        reasons = set()
        for identity in targets:
            function, target_parents, target_kind = self._syntax[identity]
            parameters = function.args.posonlyargs + function.args.args
            if function.args.vararg or function.args.kwarg:
                return {'unresolved_argument_use'}
            if keyword is not None:
                names = {arg.arg for arg in parameters + function.args.kwonlyargs}
                if keyword not in names:
                    return {'unresolved_argument_use'}
                name = keyword
            else:
                offset = int(isinstance(call.func, ast.Attribute))
                index = call.args.index(child) + offset
                if index >= len(parameters):
                    return {'unresolved_argument_use'}
                name = parameters[index].arg
            if name in self._local_values[id(function)][1]:
                return {'rebound_value'}
            reasons.update(self._local_use(function, name, target_parents, target_kind,
                                           threaded or self._threaded(call, parents, kind), visiting))
        return reasons

    def _use(self, node, parents, kind, outer_threaded=False, visiting=frozenset()):
        child = node
        while (parent := parents.get(id(child))) is not None:
            if isinstance(parent, (ast.If, ast.IfExp)) and child is parent.test:
                branches = parent.body + parent.orelse if isinstance(parent, ast.If) else (parent.body, parent.orelse)
                if uniform_range_schedule(parent, kind, self.schedule_functions, self.scalar_functions):
                    return {"prepare_condition"}
                if any(self._structural(branch, kind, parents) for branch in branches):
                    return {"structural_branch"}
                threaded = outer_threaded or self._threaded(parent, parents, kind)
                if not threaded and any(
                    isinstance(part, (ast.For, ast.While))
                    or isinstance(part, ast.Call)
                    and getattr(part.func, "attr", getattr(part.func, "id", "")) in self.execution_functions
                    for branch in branches
                    for part in ast.walk(branch)
                ):
                    return {"parallel_structure"}
                # A scalar conditional used to construct a type/block still
                # belongs to the producer, even when its two leaves are ints.
                outer = parent
                while (outer := parents.get(id(outer))) is not None and not isinstance(outer, ast.stmt):
                    if isinstance(outer, ast.Call) and kind(outer.func) != "static":
                        return {"structural_expression"}
                return {"prepare_condition"}
            if isinstance(parent, (ast.BoolOp, ast.UnaryOp, ast.Compare, ast.BinOp)):
                child = parent
                continue
            if isinstance(parent, ast.Call) and kind(parent.func) == "static":
                child = parent
                continue
            if isinstance(parent, ast.keyword):
                child = parent
                continue
            if isinstance(parent, ast.Call):
                return self._argument_use(parent, child, parents, kind, outer_threaded, visiting)
            if isinstance(parent, ast.Assign):
                function = parent
                while function is not None and not isinstance(function, ast.FunctionDef):
                    function = parents.get(id(function))
                if function is not None:
                    alias = self._local_values[id(function)][2].get(id(parent))
                    if alias is not None:
                        return self._local_use(function, alias, parents, kind, outer_threaded, visiting)
            return {"non_condition_use"}
        return {"non_condition_use"}

    def classify(self, schema, paths, identity=None):
        """Read-only facts; useful before allocating a producer witness."""
        protected = set()
        for item in schema.values():
            for key in ("conditions", "choice_guards", "shape_guards", "dtype_guards", "source_guards"):
                protected.update(item.get(key, {}))
            for key in ("shape_policy", "dtype_policy", "source_policy"):
                if key in item:
                    protected.add(item[key]["condition"])
            for choice in item.get("owner_alternatives", ()):
                protected.update(choice.get("conditions", {}))
        facts = {}
        uses = self.entry_uses(identity) if identity else self.uses
        for name, item in schema.items():
            if item["kind"] != "static" or item.get("dynamic_storage_condition"):
                continue
            choices = item["choices"]
            path = paths[name]
            reasons = uses.get(path[-1], {"unproven_use"})
            if len(choices) == 1:
                # A frozen capability has no combination cost. Keeping it at
                # build time also keeps unsupported/dead branches unreachable;
                # turning it into a runtime value would capture both arms.
                reason = "single_choice"
            elif name in protected or item.get("conditions"):
                reason = "resource_or_layout"
            elif not all(v[0] in ("bool", "int") and -(2**31) <= v[1] < 2**31 for v in choices):
                reason = "type_or_non_i32_value"
            elif reasons != {"prepare_condition"}:
                reason = ",".join(sorted(reasons - {"prepare_condition"}))
            else:
                reason = "prepare_condition"
            facts[name] = reason
        return facts

    def project(self, environment, schema, kernel):
        identity = kernel.func.__module__ + "." + kernel.func.__qualname__
        # Retain the environment identity for this build. An integer id alone
        # could be recycled while visiting another owner's layout alternatives.
        key = environment, identity
        cached = self._contexts.get(key)
        if cached is not None:
            return cached
        from taichi_forge.types import primitive_types
        from ._component import RuntimeValue

        facts = self.classify(schema, environment._paths, identity)
        declarations, projected = dict(environment.declarations), dict(schema)
        for name, reason in facts.items():
            if reason == "prepare_condition":
                declarations[name] = RuntimeValue(primitive_types.i32, affects_plan=True)
                projected[name] = {**schema[name], "prepare_static": True}
        result = copy.copy(environment)
        result._set_declarations(declarations)
        self._contexts[key] = result, projected, facts
        return result, projected, facts
