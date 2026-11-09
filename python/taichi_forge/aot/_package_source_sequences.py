"""Prove finite resource-path inventories without evaluating application data.

Only local sequence construction and literal path operations are projected.
Conditions contribute possible sources, not combinations or model snapshots.
Original Host code still chooses the collection and executes its control flow.
"""

import ast
import copy
import inspect
import operator

from ._package_factories import _source_value


class UnprovenSourceSequence(ValueError):
    pass


class Sequence:
    def __init__(self, values, *, mutable=False):
        self.values, self.mutable = list(values), mutable


class SourceSequences:
    def __init__(self, sources, imported):
        self.functions = {}
        for module, _, _, _, tree, *_ in sources:
            namespace = vars(imported[module])
            for node in tree.body:
                value = namespace.get(getattr(node, "name", ""))
                if (
                    isinstance(node, ast.FunctionDef)
                    and inspect.isfunction(value)
                    and not node.decorator_list
                    and value.__module__ == module
                    and value.__qualname__ == node.name
                    and value.__code__.co_firstlineno == node.lineno
                ):
                    self.functions[value] = (node, namespace)

    def value(self, node, environment, namespace, active=()):
        if isinstance(node, ast.Constant) and type(node.value) in (str, int, bool, type(None)):
            return copy.deepcopy(node)
        if isinstance(node, ast.Name):
            if node.id in environment:
                return environment[node.id]
            raise UnprovenSourceSequence("Inventory local has no proven value: " + node.id)
        if isinstance(node, ast.Attribute):
            parent = self.value(node.value, environment, namespace, active)
            if not isinstance(parent, ast.AST):
                raise UnprovenSourceSequence("Inventory attributes require a forwarded resource path")
            return ast.Attribute(parent, node.attr, ast.Load())
        if isinstance(node, (ast.Tuple, ast.List)):
            return Sequence(
                (self.value(n, environment, namespace, active) for n in node.elts), mutable=isinstance(node, ast.List)
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = (self.value(n, environment, namespace, active) for n in (node.left, node.right))
            if (
                isinstance(left, ast.Constant)
                and isinstance(right, ast.Constant)
                and type(left.value) is type(right.value) is str
            ):
                return ast.Constant(left.value + right.value)
            raise UnprovenSourceSequence("Inventory paths require literal string concatenation")
        if isinstance(node, ast.BoolOp):
            return ast.BoolOp(
                node.op, [self.scalar(self.value(n, environment, namespace, active)) for n in node.values]
            )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return ast.UnaryOp(node.op, self.scalar(self.value(node.operand, environment, namespace, active)))
        if isinstance(node, ast.Compare):
            return ast.Compare(
                self.scalar(self.value(node.left, environment, namespace, active)),
                node.ops,
                [self.scalar(self.value(n, environment, namespace, active)) for n in node.comparators],
            )
        if isinstance(node, (ast.GeneratorExp, ast.ListComp)):
            if len(node.generators) != 1 or node.generators[0].is_async or node.generators[0].ifs:
                raise UnprovenSourceSequence("Inventory comprehension needs one finite sequence without filters")
            generator = node.generators[0]
            values = self.sequence(self.value(generator.iter, environment, namespace, active))
            result = []
            for value in values.values:
                local = dict(environment)
                self.bind(generator.target, value, local)
                result.append(self.value(node.elt, local, namespace, active))
            return Sequence(result, mutable=isinstance(node, ast.ListComp))
        if isinstance(node, ast.Call) and not node.keywords:
            from ._resource_effects import written_resources
            from taichi_forge import aot

            if _source_value(node.func, namespace) is written_resources or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "written_resources"
                and _source_value(node.func.value, namespace) is aot
                and "written_resources" not in vars(aot)
            ):
                self.unshadowed(node.func, environment)
                if len(node.args) != 2:
                    raise UnprovenSourceSequence("Write inventory requires an owner and its recording method")
                owner = self.scalar(self.value(node.args[0], environment, namespace, active))
                record = self.scalar(self.value(node.args[1], environment, namespace, active))
                if not isinstance(record, ast.Attribute) or ast.dump(record.value) != ast.dump(owner):
                    raise UnprovenSourceSequence("Write inventory requires the same owner's recording method")
                # The owner's type may be resolved only after constructor
                # forwarding. Carry a private proof to that existing pass;
                # never execute the recording function to discover a domain.
                source = copy.deepcopy(owner)
                source._package_written_resource_inventory = record.attr
                return Sequence([source])
            if isinstance(node.func, ast.Name) and node.func.id in ("tuple", "zip"):
                import builtins

                builtin = getattr(builtins, node.func.id)
                supplied = namespace.get("__builtins__", vars(builtins))
                supplied = supplied if isinstance(supplied, dict) else vars(supplied)
                if (
                    node.func.id in environment
                    or namespace.get(node.func.id, builtin) is not builtin
                    or supplied.get(node.func.id) is not builtin
                ):
                    raise UnprovenSourceSequence("Inventory builtins must retain their original bindings")
                if node.func.id == "tuple" and len(node.args) == 1:
                    return Sequence(self.sequence(self.value(node.args[0], environment, namespace, active)).values)
                if node.func.id == "zip" and len(node.args) == 1 and isinstance(node.args[0], ast.Starred):
                    rows = self.sequence(self.value(node.args[0].value, environment, namespace, active)).values
                    return Sequence(Sequence(items) for items in zip(*(self.sequence(row).values for row in rows)))
            if isinstance(node.func, ast.Call) and len(node.args) == 1:
                selector = node.func
                self.unshadowed(selector.func, environment)
                if (
                    _source_value(selector.func, namespace) is operator.attrgetter
                    and len(selector.args) == 1
                    and not selector.keywords
                ):
                    name = self.value(selector.args[0], environment, namespace, active)
                    if (
                        not isinstance(name, ast.Constant)
                        or type(name.value) is not str
                        or not all(part.isidentifier() for part in name.value.split("."))
                    ):
                        raise UnprovenSourceSequence("Inventory attrgetter requires a literal resource path")
                    result = self.scalar(self.value(node.args[0], environment, namespace, active))
                    for part in name.value.split("."):
                        result = ast.Attribute(result, part, ast.Load())
                    return result
            self.unshadowed(node.func, environment)
            target = _source_value(node.func, namespace)
            if inspect.isfunction(target) and target in self.functions:
                if target in active:
                    raise UnprovenSourceSequence("Recursive source inventories are not finite declarations")
                arguments = [self.value(arg, environment, namespace, active) for arg in node.args]
                bound = inspect.signature(target).bind(*arguments)
                # Defaults cannot silently introduce runtime objects into the proof.
                if len(bound.arguments) != len(inspect.signature(target).parameters):
                    raise UnprovenSourceSequence("Inventory helpers require explicit arguments")
                definition, child_namespace = self.functions[target]
                result = self.body(definition.body, dict(bound.arguments), child_namespace, (*active, target))
                if not self.immutable(result):
                    raise UnprovenSourceSequence("Inventory helpers must return an immutable source tuple")
                return result
        raise UnprovenSourceSequence("Unproven source inventory expression")

    @staticmethod
    def unshadowed(node, environment):
        while isinstance(node, ast.Attribute):
            node = node.value
        if not isinstance(node, ast.Name) or node.id in environment:
            raise UnprovenSourceSequence("Inventory call requires an unshadowed definition")

    @classmethod
    def immutable(cls, value):
        return (
            isinstance(value, Sequence)
            and not value.mutable
            and all(
                cls.immutable(item) if isinstance(item, Sequence) else isinstance(item, ast.AST)
                for item in value.values
            )
        )

    @staticmethod
    def scalar(value):
        if not isinstance(value, ast.AST):
            raise UnprovenSourceSequence("Inventory predicate/path requires a scalar expression")
        return value

    @staticmethod
    def sequence(value):
        if not isinstance(value, Sequence):
            raise UnprovenSourceSequence("Inventory iteration requires a finite source sequence")
        return value

    def bind(self, target, value, environment, *, allow_members=False):
        if isinstance(target, ast.Name):
            environment[target.id] = value
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            values = self.sequence(value).values
            if len(values) == len(target.elts):
                for child, item in zip(target.elts, values):
                    self.bind(child, item, environment, allow_members=allow_members)
                return
        if (
            allow_members
            and isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            return
        raise UnprovenSourceSequence("Inventory unpacking differs from its declared tuple")

    def body(self, statements, environment, namespace, active, *, branch=False):
        for node in statements:
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and not branch:
                self.bind(node.targets[0], self.value(node.value, environment, namespace, active), environment)
            elif isinstance(node, ast.For) and not node.orelse:
                values = self.sequence(self.value(node.iter, environment, namespace, active))
                for value in values.values:
                    local = dict(environment)
                    self.bind(node.target, value, local)
                    if self.body(node.body, local, namespace, active, branch=branch) is not None:
                        raise UnprovenSourceSequence("Inventory loops cannot return early")
            elif isinstance(node, ast.If):
                self.scalar(self.value(node.test, environment, namespace, active))
                for arm in (node.body, node.orelse):
                    if self.body(arm, dict(environment), namespace, active, branch=True) is not None:
                        raise UnprovenSourceSequence("Inventory branches cannot return early")
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if (
                    not isinstance(call.func, ast.Attribute)
                    or not isinstance(call.func.value, ast.Name)
                    or len(call.args) != 1
                    or call.keywords
                    or call.func.attr not in ("append", "extend")
                ):
                    raise UnprovenSourceSequence("Inventory statements may only append local source declarations")
                collection = self.sequence(environment.get(call.func.value.id))
                if not collection.mutable:
                    raise UnprovenSourceSequence("Inventory cannot mutate a forwarded immutable sequence")
                value = self.value(call.args[0], environment, namespace, active)
                collection.values.extend([value] if call.func.attr == "append" else self.sequence(value).values)
            elif isinstance(node, ast.Return) and not branch:
                if node is not statements[-1]:
                    raise UnprovenSourceSequence("Inventory requires a final return")
                return self.value(node.value, environment, namespace, active)
            else:
                raise UnprovenSourceSequence("Inventory has an unproven statement or side effect")

    def constructor_aliases(self, constructor, namespace):
        environment = {
            arg.arg: ast.Name(arg.arg, ast.Load())
            for arg in (*constructor.args.posonlyargs, *constructor.args.args, *constructor.args.kwonlyargs)
        }
        writes = {}
        for node in ast.walk(constructor):
            if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                writes[node.id] = writes.get(node.id, 0) + 1
        result = {}
        for node in constructor.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            names = [n.id for n in ast.walk(target) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]
            if not names or any(writes.get(name) != 1 for name in names):
                continue
            try:
                value = self.value(node.value, environment, namespace)
                local = dict(environment)
                self.bind(target, value, local, allow_members=True)
            except (UnprovenSourceSequence, TypeError):
                continue
            environment.update(local)
            for name in names:
                value = local[name]
                if self.immutable(value):
                    result[name] = self.syntax(value)
        return result

    def syntax(self, value):
        if isinstance(value, Sequence):
            return ast.Tuple([self.syntax(item) for item in value.values], ast.Load())
        return copy.deepcopy(value)
