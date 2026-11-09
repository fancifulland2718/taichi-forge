"""Prove finite resource-path inventories without evaluating application data.

Only local sequence construction and literal path operations are projected.
Conditions contribute possible sources, not combinations or model snapshots.
Original Host code still chooses the collection and executes its control flow.
"""

import ast
import builtins
import copy
import inspect
import itertools
import operator

from ._package_factories import _source_value


class UnprovenSourceSequence(ValueError):
    pass


class _UnavailableSource(UnprovenSourceSequence):
    """A source-only branch terminates by rejecting the configuration."""


class Sequence:
    def __init__(self, values, *, mutable=False, domain=False):
        self.values, self.mutable, self.domain = list(values), mutable, domain


class SourceSequences:
    def __init__(self, sources, imported, *, inventory=False):
        self.inventory = inventory
        self.choices = itertools.count(1)
        self.resolve_method = None
        self.resolve_cached = None
        self.resolve_namespace = None
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

    def choice(self, predicate):
        return predicate, next(self.choices)

    @staticmethod
    def source_key(value):
        if isinstance(value, Sequence):
            return (value.mutable, value.domain, tuple(SourceSequences.source_key(item) for item in value.values))
        return (ast.dump(value), tuple((getattr(part, '_package_source_choice', None),
                                       getattr(part, '_package_unknown_iteration', False),
                                       getattr(part, '_package_source_callable', None)) for part in ast.walk(value)))

    def join(self, left, right, choice, memo=None):
        """Merge sources without inventing an ordered concatenation of branches."""
        memo = {} if memo is None else memo
        key = id(left), id(right)
        if key in memo:
            return memo[key]
        if isinstance(left, Sequence) and isinstance(right, Sequence):
            if left.mutable != right.mutable:
                raise UnprovenSourceSequence('Inventory branches disagree on sequence mutability')
            if left.mutable:
                # A branch may append to a local list, but changing which local
                # aliases that list needs path-sensitive mutation analysis.
                # Do not lose writes through an alias after joining the arms.
                for side, source, other in (('left', left, right), ('right', right, left)):
                    alias = side, id(source)
                    if alias in memo and memo[alias] != id(other):
                        raise UnprovenSourceSequence('Inventory branches change mutable sequence aliases')
                    memo[alias] = id(other)
            if left.mutable or left.domain or right.domain:
                if any(isinstance(item, Sequence) and not self.immutable(item)
                       for item in (*left.values, *right.values)):
                    raise UnprovenSourceSequence('Inventory branches require immutable sequence elements')
                values = {self.source_key(item): item for item in (*left.values, *right.values)}
                result = Sequence(values.values(), mutable=left.mutable, domain=True)
            elif len(left.values) == len(right.values):
                result = Sequence(self.join(a, b, choice, memo) for a, b in zip(left.values, right.values))
            else:
                values = {self.source_key(item): item for item in (*left.values, *right.values)}
                result = Sequence(values.values(), domain=True)
        elif self.immutable(left) and isinstance(right, ast.AST):
            return self.join(self.syntax(left), right, choice, memo)
        elif isinstance(left, ast.AST) and self.immutable(right):
            return self.join(left, self.syntax(right), choice, memo)
        elif isinstance(left, ast.AST) and isinstance(right, ast.AST):
            if self.source_key(left) == self.source_key(right):
                result = left
            else:
                result = ast.IfExp(copy.deepcopy(choice[0]), left, right)
                result._package_source_choice = choice[1]
        else:
            raise UnprovenSourceSequence('Inventory branches disagree on source structure')
        memo[key] = result
        return result

    def member(self, parent, name):
        if isinstance(name, ast.IfExp) and hasattr(name, '_package_source_choice'):
            result = ast.IfExp(name.test, self.member(parent, name.body), self.member(parent, name.orelse))
            result._package_source_choice = name._package_source_choice
            return result
        if not isinstance(name, ast.Constant) or type(name.value) is not str:
            raise UnprovenSourceSequence('Inventory getattr requires a literal member name')
        return ast.Attribute(parent, name.value, ast.Load())

    @classmethod
    def known_condition(cls, expression):
        if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
            answer = cls.known_condition(expression.operand)
            return None if answer is None else not answer
        if isinstance(expression, ast.Compare) and len(expression.ops) == 1:
            operations = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.In: lambda a, b: a in b,
                          ast.NotIn: lambda a, b: a not in b, ast.Is: operator.is_, ast.IsNot: operator.is_not}
            operation = operations.get(type(expression.ops[0]))
            def alternatives(node):
                if isinstance(node, ast.IfExp):
                    return alternatives(node.body) + alternatives(node.orelse)
                return [ast.literal_eval(node)]
            if operation is not None:
                try:
                    left, right = alternatives(expression.left), alternatives(expression.comparators[0])
                    if isinstance(expression.ops[0], (ast.Is, ast.IsNot)) and not all(
                            item is None or type(item) is bool for item in (*left, *right)):
                        return None
                    answers = {operation(a, b) for a in left for b in right}
                    if len(answers) == 1:
                        return answers.pop()
                except (ValueError, TypeError):
                    pass
        try:
            return bool(ast.literal_eval(expression))
        except (ValueError, TypeError):
            return None

    def value(self, node, environment, namespace, active=()):
        if isinstance(node, ast.Constant) and type(node.value) in (str, int, float, bool, type(None)):
            return copy.deepcopy(node)
        if isinstance(node, ast.Name):
            if node.id in environment:
                return environment[node.id]
            target = namespace.get(node.id)
            if self.inventory and type(target) in (str, int, bool) and node.id in namespace:
                return ast.Constant(target)
            if self.inventory and inspect.isfunction(target) and target in self.functions:
                result = copy.deepcopy(node)
                result._package_source_callable = target
                return result
            raise UnprovenSourceSequence("Inventory local has no proven value: " + node.id)
        if isinstance(node, ast.Attribute):
            parent = self.value(node.value, environment, namespace, active)
            if not isinstance(parent, ast.AST):
                raise UnprovenSourceSequence("Inventory attributes require a forwarded resource path")
            expression = ast.Attribute(parent, node.attr, ast.Load())
            if self.inventory:
                expression._package_source_active = active
            return expression
        if self.inventory and isinstance(node, ast.Lambda):
            arguments = node.args
            if (arguments.posonlyargs or arguments.args or arguments.kwonlyargs
                    or arguments.vararg or arguments.kwarg):
                raise UnprovenSourceSequence('Inventory callback requires explicit immutable captures')
            projection = self
            class Captures(ast.NodeTransformer):
                def visit_Name(self, part):
                    if isinstance(part.ctx, ast.Load) and part.id in environment:
                        return ast.copy_location(projection.syntax(environment[part.id]), part)
                    target = namespace.get(part.id)
                    if inspect.isfunction(target) and target in projection.functions:
                        part._package_source_callable = target
                    return part
            return Captures().visit(copy.deepcopy(node))
        if self.inventory and isinstance(node, ast.Subscript):
            parent = self.scalar(self.value(node.value, environment, namespace, active))
            key = self.value(node.slice, environment, namespace, active)
            if not isinstance(key, ast.Constant) or type(key.value) not in (str, int):
                raise UnprovenSourceSequence('Inventory metadata lookup requires a literal key')
            return ast.Subscript(parent, key, ast.Load())
        if isinstance(node, (ast.Tuple, ast.List)):
            return Sequence(
                (self.value(n, environment, namespace, active) for n in node.elts), mutable=isinstance(node, ast.List)
            )
        if self.inventory and isinstance(node, ast.IfExp):
            predicate = self.scalar(self.value(node.test, environment, namespace, active))
            chosen = self.known_condition(predicate)
            if chosen is not None:
                return self.value(node.body if chosen else node.orelse, environment, namespace, active)
            return self.join(self.value(node.body, environment, namespace, active),
                             self.value(node.orelse, environment, namespace, active), self.choice(predicate))
        if isinstance(node, ast.BinOp) and (self.inventory or isinstance(node.op, ast.Add)):
            left, right = (self.value(n, environment, namespace, active) for n in (node.left, node.right))
            if (
                isinstance(node.op, ast.Add)
                and isinstance(left, ast.Constant)
                and isinstance(right, ast.Constant)
                and type(left.value) is type(right.value) is str
            ):
                return ast.Constant(left.value + right.value)
            if self.inventory and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
                return ast.BinOp(self.scalar(left), node.op, self.scalar(right))
            raise UnprovenSourceSequence("Inventory paths require literal string concatenation")
        if isinstance(node, ast.BoolOp):
            return ast.BoolOp(
                node.op, [self.scalar(self.value(n, environment, namespace, active)) for n in node.values]
            )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return ast.UnaryOp(node.op, self.scalar(self.value(node.operand, environment, namespace, active)))
        if isinstance(node, ast.Compare):
            def operand(expression):
                value = self.value(expression, environment, namespace, active)
                return self.syntax(value) if self.inventory and self.immutable(value) else self.scalar(value)

            return ast.Compare(
                operand(node.left),
                node.ops,
                [operand(n) for n in node.comparators],
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
            return Sequence(result, mutable=isinstance(node, ast.ListComp), domain=values.domain)
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
            if isinstance(node.func, ast.Name) and node.func.id == 'getattr' and len(node.args) in (2, 3):
                self.unshadowed(node.func, environment)
                supplied = namespace.get('__builtins__', vars(builtins))
                supplied = supplied if isinstance(supplied, dict) else vars(supplied)
                if (namespace.get('getattr', builtins.getattr) is not builtins.getattr
                        or supplied.get('getattr') is not builtins.getattr):
                    raise UnprovenSourceSequence('Inventory getattr must retain its original binding')
                parent = self.scalar(self.value(node.args[0], environment, namespace, active))
                name = self.value(node.args[1], environment, namespace, active)
                if len(node.args) == 3:
                    self.scalar(self.value(node.args[2], environment, namespace, active))
                return self.member(parent, name)
            if isinstance(node.func, ast.Name) and node.func.id in (("tuple", "list", "zip") if self.inventory else ("tuple", "zip")):
                builtin = getattr(builtins, node.func.id)
                supplied = namespace.get("__builtins__", vars(builtins))
                supplied = supplied if isinstance(supplied, dict) else vars(supplied)
                if (
                    node.func.id in environment
                    or namespace.get(node.func.id, builtin) is not builtin
                    or supplied.get(node.func.id) is not builtin
                ):
                    raise UnprovenSourceSequence("Inventory builtins must retain their original bindings")
                if node.func.id in ("tuple", "list") and len(node.args) == 1:
                    value = self.sequence(self.value(node.args[0], environment, namespace, active))
                    return Sequence(value.values, mutable=node.func.id == 'list', domain=value.domain)
                if node.func.id == "zip" and len(node.args) == 1 and isinstance(node.args[0], ast.Starred):
                    sequence = self.sequence(self.value(node.args[0].value, environment, namespace, active))
                    if sequence.domain or any(self.sequence(row).domain for row in sequence.values):
                        raise UnprovenSourceSequence('Inventory source domains do not prove zip positions')
                    rows = sequence.values
                    return Sequence(Sequence(items) for items in zip(*(self.sequence(row).values for row in rows)))
            if self.inventory and isinstance(node.func, ast.Name) and node.func.id == 'bool' and len(node.args) == 1:
                supplied = namespace.get('__builtins__', vars(builtins))
                supplied = supplied if isinstance(supplied, dict) else vars(supplied)
                if ('bool' in environment or namespace.get('bool', builtins.bool) is not builtins.bool
                        or supplied.get('bool') is not builtins.bool):
                    raise UnprovenSourceSequence('Inventory bool must retain its original binding')
                predicate = self.scalar(self.value(node.args[0], environment, namespace, active))
                return ast.UnaryOp(ast.Not(), ast.UnaryOp(ast.Not(), predicate))
            if self.inventory and isinstance(node.func, ast.Name) and node.func.id == 'str' and len(node.args) == 1:
                supplied = namespace.get('__builtins__', vars(builtins))
                supplied = supplied if isinstance(supplied, dict) else vars(supplied)
                if ('str' in environment or namespace.get('str', builtins.str) is not builtins.str
                        or supplied.get('str') is not builtins.str):
                    raise UnprovenSourceSequence('Inventory str must retain its original binding')
                item = self.scalar(self.value(node.args[0], environment, namespace, active))
                if isinstance(item, ast.Constant):
                    return ast.Constant(str(item.value))
                function = copy.deepcopy(node.func)
                function._package_source_callable = builtins.str
                return ast.Call(function, [item], [])
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
        if isinstance(node, ast.Call):
            if self.inventory:
                from ._package_function_inputs import _plain_record

                target = _source_value(node.func, namespace)
                if _plain_record(target) and all(k.arg is not None for k in node.keywords):
                    function = copy.deepcopy(node.func)
                    function._package_source_callable = target
                    args = [self.syntax(item, allow_domain=True) for item in self.arguments(node.args, environment, namespace, active)]
                    kwargs = [ast.keyword(k.arg, self.syntax(self.value(k.value, environment, namespace, active), allow_domain=True))
                              for k in node.keywords]
                    return ast.Call(function, args, kwargs)
            receiver = None
            root = node.func
            while isinstance(root, ast.Attribute):
                root = root.value
            if self.inventory and isinstance(node.func, ast.Name) and node.func.id in environment:
                chosen = environment[node.func.id]
                arguments = self.arguments(node.args, environment, namespace, active)
                keywords = {k.arg: self.value(k.value, environment, namespace, active) for k in node.keywords}
                if None in keywords:
                    raise UnprovenSourceSequence('Inventory helpers require explicit keyword arguments')
                return self.invoke_choice(chosen, arguments, keywords, active)
            if (self.inventory and self.resolve_method is not None and isinstance(node.func, ast.Attribute)
                    and isinstance(root, ast.Name) and root.id in environment):
                receiver = self.scalar(self.value(node.func.value, environment, namespace, active))
                target = self.resolve_method(ast.Attribute(receiver, node.func.attr, ast.Load()))
                if target is None:
                    raise UnprovenSourceSequence('Inventory method requires a proven ordinary receiver definition')
            else:
                self.unshadowed(node.func, environment)
                target = _source_value(node.func, namespace)
            if inspect.isfunction(target) and target in self.functions:
                if any(k.arg is None for k in node.keywords):
                    raise UnprovenSourceSequence("Inventory helpers require explicit keyword arguments")
                arguments = self.arguments(node.args, environment, namespace, active)
                if receiver is not None:
                    arguments.insert(0, receiver)
                keywords = {k.arg: self.value(k.value, environment, namespace, active) for k in node.keywords}
                return self.invoke(target, arguments, keywords, active)
        raise UnprovenSourceSequence(f"Unproven source inventory expression: {ast.unparse(node)}")

    def arguments(self, nodes, environment, namespace, active):
        result = []
        for node in nodes:
            if isinstance(node, ast.Starred):
                values = self.sequence(self.value(node.value, environment, namespace, active))
                if values.domain:
                    raise UnprovenSourceSequence('Inventory source domain does not prove argument positions')
                result.extend(values.values)
            else:
                result.append(self.value(node, environment, namespace, active))
        return result

    def invoke_choice(self, expression, arguments, keywords, active):
        if isinstance(expression, ast.IfExp) and hasattr(expression, '_package_source_choice'):
            results = [self.invoke_choice(arm, arguments, keywords, active)
                       for arm in (expression.body, expression.orelse)]
            if all(self.immutable(result) for result in results):
                unique = {self.source_key(item): item for result in results for item in result.values}
                return Sequence(unique.values(), domain=True)
            return self.join(*results, (expression.test, expression._package_source_choice))
        target = getattr(expression, '_package_source_callable', None)
        if target not in self.functions:
            raise UnprovenSourceSequence('Inventory callable has no admitted source')
        return self.invoke(target, arguments, keywords, active)

    def invoke(self, target, arguments, keywords, active):
        if target in active:
            raise UnprovenSourceSequence("Recursive source inventories are not finite declarations")
        definition, namespace = self.functions[target]
        if self.resolve_namespace is not None:
            namespace = self.resolve_namespace(target)
        signature = inspect.signature(target)
        bound = signature.bind(*arguments, **keywords)
        for name, parameter in signature.parameters.items():
            if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                bound.arguments[name] = Sequence(bound.arguments.get(name, ()))
            elif name not in bound.arguments:
                if type(parameter.default) not in (bool, int, str, type(None)):
                    raise UnprovenSourceSequence("Inventory helpers require explicit immutable arguments")
                bound.arguments[name] = ast.Constant(parameter.default)
        try:
            project = self.generator_body if inspect.isgeneratorfunction(target) else self.body
            result = project(definition.body, dict(bound.arguments), namespace, (*active, target))
        except _UnavailableSource:
            raise
        except UnprovenSourceSequence as error:
            raise UnprovenSourceSequence(target.__qualname__ + ": " + str(error)) from error
        if not self.immutable(result) and not (self.inventory and isinstance(result, ast.AST)):
            raise UnprovenSourceSequence("Inventory helpers must return an immutable source tuple")
        return result

    def iteration(self, expression, environment, namespace, active):
        if (isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name)
                and expression.func.id == 'range' and 'range' not in environment
                and namespace.get('range', builtins.range) is builtins.range
                and 1 <= len(expression.args) <= 3 and not expression.keywords):
            supplied = namespace.get('__builtins__', vars(builtins))
            supplied = supplied if isinstance(supplied, dict) else vars(supplied)
            if supplied.get('range') is not builtins.range:
                raise UnprovenSourceSequence('Inventory range must retain its original binding')
            for argument in expression.args:
                self.scalar(self.value(argument, environment, namespace, active))
            unknown = ast.Name('_package_dynamic_iteration', ast.Load())
            unknown._package_unknown_iteration = True
            return Sequence([unknown], domain=True)
        return self.sequence(self.value(expression, environment, namespace, active))

    def join_environments(self, left, right, predicate):
        choice, memo = self.choice(predicate), {}
        return {name: self.join(left[name], right[name], choice, memo) for name in left.keys() & right.keys()}

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

    def sequence(self, value):
        if self.inventory and isinstance(value, ast.Tuple):
            return Sequence(value.elts, domain=getattr(value, '_package_source_domain', False))
        if self.inventory and self.resolve_cached is not None and isinstance(value, ast.Attribute):
            target = self.resolve_cached(value)
            if target is not None:
                return self.sequence(self.invoke(target, [value.value], {}, getattr(value, '_package_source_active', ())))
        if not isinstance(value, Sequence):
            raise UnprovenSourceSequence("Inventory iteration requires a finite source sequence")
        return value

    def refined(self, test, environment, take):
        """Narrow only explicit optional source tests; never evaluate user data."""
        result = copy.deepcopy(environment)
        if (not isinstance(test, ast.Compare) or len(test.ops) != 1
                or not isinstance(test.left, ast.Name)
                or not isinstance(test.comparators[0], ast.Constant)
                or test.comparators[0].value is not None
                or not isinstance(test.ops[0], (ast.Is, ast.IsNot))):
            return result
        name = test.left.id
        want_none = take == isinstance(test.ops[0], ast.Is)
        def narrow(value):
            if isinstance(value, ast.IfExp) and hasattr(value, '_package_source_choice'):
                left, right = narrow(value.body), narrow(value.orelse)
                if left is None:
                    return right
                if right is None:
                    return left
                return self.join(left, right, (value.test, value._package_source_choice))
            if isinstance(value, ast.Constant) and value.value is None:
                return value if want_none else None
            if self.immutable(value) or isinstance(value, ast.Tuple):
                return None if want_none else value
            return value
        if name in result:
            result[name] = narrow(result[name])
            if result[name] is None:
                return None
        return result

    def bind(self, target, value, environment, *, allow_members=False):
        if isinstance(target, ast.Name):
            environment[target.id] = value
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            sequence = self.sequence(value)
            if sequence.domain:
                raise UnprovenSourceSequence('Inventory source domains do not prove unpacking positions')
            values = sequence.values
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
        for index, node in enumerate(statements):
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and (not branch or self.inventory):
                self.bind(node.targets[0], self.value(node.value, environment, namespace, active), environment)
            elif self.inventory and isinstance(node, ast.AnnAssign) and node.value is not None:
                self.bind(node.target, self.value(node.value, environment, namespace, active), environment)
            elif isinstance(node, ast.For) and not node.orelse:
                values = (self.iteration(node.iter, environment, namespace, active) if self.inventory else
                          self.sequence(self.value(node.iter, environment, namespace, active)))
                if self.inventory and any(isinstance(part, ast.Name) and not isinstance(part.ctx, ast.Load)
                                          and part.id in environment for part in ast.walk(node)):
                    raise UnprovenSourceSequence('Inventory loop cannot rebind an existing source')
                for value in values.values:
                    local = copy.deepcopy(environment) if self.inventory else dict(environment)
                    self.bind(node.target, value, local)
                    if self.body(node.body, local, namespace, active, branch=branch) is not None:
                        raise UnprovenSourceSequence("Inventory loops cannot return early")
                    if self.inventory:
                        updated = (self.join_environments(environment, local, ast.Constant(True))
                                   if values.domain else {name: local[name] for name in environment})
                        environment.update(updated)
            elif isinstance(node, ast.If):
                predicate = self.scalar(self.value(node.test, environment, namespace, active))
                chosen = self.known_condition(predicate) if self.inventory else None
                if chosen is not None:
                    selected = self.body(node.body if chosen else node.orelse, environment, namespace, active)
                    if selected is not None:
                        return selected
                    continue
                if self.inventory and any(isinstance(part, (ast.Return, ast.Raise)) for part in ast.walk(node)):
                    # Complete each returning arm through the remaining source
                    # statements. Return values are possible elements, never a
                    # positional concatenation of mutually exclusive tuples.
                    results = []
                    for take, arm in ((True, node.body), (False, node.orelse)):
                        local = self.refined(node.test, environment, take)
                        if local is not None:
                            try:
                                results.append(self.body(arm + statements[index + 1:], local, namespace, active))
                            except _UnavailableSource:
                                pass
                    if not results:
                        raise _UnavailableSource('All source branches reject the configuration')
                    if len(results) == 1:
                        return results[0]
                    return self.join(*results, self.choice(predicate))
                branches = []
                for take, arm in ((True, node.body), (False, node.orelse)):
                    local = self.refined(node.test, environment, take) if self.inventory else dict(environment)
                    if local is None:
                        continue
                    if self.body(arm, local, namespace, active, branch=True) is not None:
                        raise UnprovenSourceSequence("Inventory branches cannot return early")
                    branches.append(local)
                if self.inventory:
                    joined = branches[0] if len(branches) == 1 else self.join_environments(*branches, predicate)
                    environment.clear()
                    environment.update(joined)
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
                if call.func.attr == 'extend':
                    collection.domain |= self.sequence(value).domain
            elif isinstance(node, ast.Return) and (not branch or self.inventory):
                if not self.inventory and node is not statements[-1]:
                    raise UnprovenSourceSequence("Inventory requires a final return")
                return self.value(node.value, environment, namespace, active)
            elif self.inventory and isinstance(node, ast.Raise):
                call = node.exc
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                        and call.func.id not in environment and not call.keywords and node.cause is None
                        and all(isinstance(arg, ast.Constant) for arg in call.args)):
                    supplied = namespace.get('__builtins__', vars(builtins))
                    supplied = supplied if isinstance(supplied, dict) else vars(supplied)
                    target = namespace.get(call.func.id, supplied.get(call.func.id))
                    if target in (ValueError, TypeError, RuntimeError):
                        raise _UnavailableSource('Source branch rejects the configuration')
                raise UnprovenSourceSequence('Inventory exception has an unproven constructor or argument')
            elif self.inventory and isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (item.asname or (item.name if isinstance(node, ast.ImportFrom) else item.name.split('.')[0])
                         for item in node.names)
                if any(name in environment for name in names):
                    raise UnprovenSourceSequence('Inventory import cannot rebind a source')
            else:
                raise UnprovenSourceSequence(
                    f"Inventory has an unproven {type(node).__name__} or side effect at line {node.lineno}")

    def generator_body(self, statements, environment, namespace, active):
        """Project yielded sources, never the number of runtime iterations."""
        result = []
        for node in statements:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                self.bind(node.targets[0], self.value(node.value, environment, namespace, active), environment)
            elif isinstance(node, ast.Expr) and isinstance(node.value, (ast.Yield, ast.YieldFrom)):
                value = self.value(node.value.value, environment, namespace, active)
                result.extend(self.sequence(value).values if isinstance(node.value, ast.YieldFrom) else [value])
            elif isinstance(node, ast.If):
                predicate = self.scalar(self.value(node.test, environment, namespace, active))
                chosen = self.known_condition(predicate)
                if chosen is not None:
                    result.extend(self.generator_body(node.body if chosen else node.orelse, environment, namespace, active).values)
                    continue
                branches = []
                for branch in (node.body, node.orelse):
                    local = copy.deepcopy(environment)
                    result.extend(self.generator_body(branch, local, namespace, active).values)
                    branches.append(local)
                joined = self.join_environments(*branches, predicate)
                environment.clear()
                environment.update(joined)
            elif isinstance(node, ast.For) and not node.orelse:
                inputs = set(inspect.signature(active[-1]).parameters) if active else set(environment)
                if any(part.id in inputs for part in ast.walk(node.target) if isinstance(part, ast.Name)):
                    raise UnprovenSourceSequence('Generator iteration cannot rebind a forwarded input')
                values = self.iteration(node.iter, environment, namespace, active).values
                for value in values:
                    local = dict(environment)
                    self.bind(node.target, value, local)
                    result.extend(self.generator_body(node.body, local, namespace, active).values)
            elif isinstance(node, ast.Return) and node.value is None:
                break
            elif isinstance(node, ast.Continue):
                break
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (item.asname or (item.name if isinstance(node, ast.ImportFrom) else item.name.split('.')[0])
                         for item in node.names)
                if any(name in environment for name in names):
                    raise UnprovenSourceSequence('Generator import cannot rebind a forwarded input')
                continue  # Imports are resolved by the admitted source namespace.
            elif isinstance(node, ast.Pass):
                continue
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            else:
                raise UnprovenSourceSequence(
                    f'Generator has an unproven {type(node).__name__} at line {node.lineno}')
        # The source domain, not the repeated dispatch plan, is the proof input.
        unique = {}
        for value in result:
            unique.setdefault(self.source_key(value), value)
        return Sequence(unique.values())

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

    def syntax(self, value, *, allow_domain=False):
        if isinstance(value, Sequence):
            if value.domain and not allow_domain:
                raise UnprovenSourceSequence('Inventory source domain is not an ordered tuple value')
            result = ast.Tuple([self.syntax(item, allow_domain=allow_domain) for item in value.values], ast.Load())
            if value.domain:
                result._package_source_domain = True
            return result
        return copy.deepcopy(value)
