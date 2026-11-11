"""Classify package configuration before visiting its finite code domain.

Only producer-side syntax and declaration facts are inspected here. Consumers
receive scalar contracts and compiled control flow, never these syntax trees.
Unproven structural uses keep the ordinary finite specialization contract.
"""

import ast
import copy


class StaticPreparation:
    def __init__(self, sources):
        self.uses = {}
        self._contexts = {}
        self._entry_uses = {}
        self._functions = {}
        self._names = {}
        functions = []
        for module, _, _, _, tree, kind, _, _ in sources:
            tree_parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
            for function in ast.walk(tree):
                if not isinstance(function, ast.FunctionDef) or not any(
                    kind(d) in ("kernel", "func", "real_func", "pyfunc") for d in function.decorator_list
                ):
                    continue
                names, ancestor = [function.name], function
                while (ancestor := tree_parents.get(id(ancestor))) is not None:
                    if isinstance(ancestor, ast.ClassDef):
                        names.insert(0, ancestor.name)
                identity = module + "." + ".".join(names)
                functions.append((identity, function, kind))
                self._names.setdefault(function.name, set()).add(identity)
        # Inline helpers at kernel scope can contribute parallel tasks. Follow
        # their ordinary call names conservatively before moving a condition;
        # the absence of loop_config at the call site is not a legality proof.
        self.execution_functions = set()
        calls = {}
        for _, function, kind in functions:
            if any(kind(d) == "real_func" for d in function.decorator_list):
                continue
            called = calls.setdefault(function.name, set())
            for node in ast.walk(function):
                if isinstance(node, (ast.For, ast.While)) or (
                    isinstance(node, ast.Attribute) and node.attr in ("loop_config", "SharedArray", "simt")
                ):
                    self.execution_functions.add(function.name)
                if isinstance(node, ast.Call):
                    called.add(getattr(node.func, "attr", getattr(node.func, "id", "")))
        while True:
            extra = {name for name, called in calls.items() if called & self.execution_functions}
            if extra <= self.execution_functions:
                break
            self.execution_functions.update(extra)
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
                        for context in uses.values():
                            context.setdefault(node.args[1].value, set()).add("reflective_use")
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                    for threaded, context in uses.items():
                        context.setdefault(node.attr, set()).add(self._use(node, parents, kind, threaded))
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

    @staticmethod
    def _structural(node, kind):
        # These operations define types, Python objects, or execution structure.
        # Their branch alternatives cannot be represented by a scalar IR phi.
        for part in ast.walk(node):
            if isinstance(part, (ast.Return, ast.Break, ast.Continue, ast.Raise)):
                return True
            if isinstance(part, ast.Attribute) and (
                part.attr in ("loop_config", "SharedArray", "Vector", "Matrix", "field", "ndarray")
                or (kind(part) or "").startswith("types.")
                or kind(part) in ("i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f16", "f32", "f64")
            ):
                return True
            if isinstance(part, ast.Assign) and any(isinstance(t, ast.Name) for t in part.targets):
                # Local Python aliases can select functions/resources/types.
                # Without a typed proof, preserve their original static scope.
                if isinstance(part.value, (ast.Name, ast.Attribute, ast.Tuple, ast.List, ast.Dict)):
                    return True
        return False

    def _use(self, node, parents, kind, outer_threaded=False):
        child = node
        while (parent := parents.get(id(child))) is not None:
            if isinstance(parent, (ast.If, ast.IfExp)) and child is parent.test:
                branches = parent.body + parent.orelse if isinstance(parent, ast.If) else (parent.body, parent.orelse)
                if any(self._structural(branch, kind) for branch in branches):
                    return "structural_branch"
                threaded = outer_threaded or self._threaded(parent, parents, kind)
                if not threaded and any(
                    isinstance(part, (ast.For, ast.While))
                    or isinstance(part, ast.Call)
                    and getattr(part.func, "attr", getattr(part.func, "id", "")) in self.execution_functions
                    for branch in branches
                    for part in ast.walk(branch)
                ):
                    return "parallel_structure"
                # A scalar conditional used to construct a type/block still
                # belongs to the producer, even when its two leaves are ints.
                outer = parent
                while (outer := parents.get(id(outer))) is not None and not isinstance(outer, ast.stmt):
                    if isinstance(outer, ast.Call) and kind(outer.func) != "static":
                        return "structural_expression"
                return "prepare_condition"
            if isinstance(parent, (ast.BoolOp, ast.UnaryOp, ast.Compare, ast.BinOp)):
                child = parent
                continue
            if isinstance(parent, ast.Call) and kind(parent.func) == "static":
                child = parent
                continue
            return "non_condition_use"
        return "non_condition_use"

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
            if name in protected or item.get("conditions"):
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
