"""Normalize fixed Struct member factories without executing application code."""

import ast
import copy
import hashlib
import inspect
import textwrap

from ._package_construction import construction_functions, ordinary_owner_lookup
from ._package_types import qualified_kind


def normalize_type_member_factories(tree, namespace, kind, module, fail):
    facts = []
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        owner = namespace.get(cls.name)

        def members(expression):
            if isinstance(expression, ast.Dict):
                return expression
            if not (
                isinstance(expression, ast.Call)
                and isinstance(expression.func, ast.Attribute)
                and isinstance(expression.func.value, ast.Name)
                and expression.func.value.id == "self"
                and ordinary_owner_lookup(owner)
            ):
                fail(module, expression, "Type member expansion requires a fixed dictionary or ordinary method")
            name = expression.func.attr
            if any(
                isinstance(n, ast.Attribute)
                and isinstance(n.value, ast.Name)
                and n.value.id == "self"
                and n.attr == name
                and not isinstance(n.ctx, ast.Load)
                for n in ast.walk(cls)
            ):
                fail(module, expression, "Type member method cannot be replaced on the owner")
            method = inspect.getattr_static(owner, name, None)
            if not inspect.isfunction(method) or "__wrapped__" in vars(method):
                fail(module, expression, "Type member factory requires an ordinary source method")
            try:
                source = textwrap.dedent(inspect.getsource(method))
                definition = ast.parse(source).body[0]
                bound = inspect.signature(method).bind(
                    ast.Name("self", ast.Load()), *expression.args, **{k.arg: k.value for k in expression.keywords}
                )
            except (TypeError, ValueError, OSError) as error:
                fail(module, expression, f"Type member factory signature is unproven: {error}")
            body = [
                n
                for n in definition.body
                if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and type(n.value.value) is str)
            ]
            if (
                definition.decorator_list
                or len(body) != 1
                or not isinstance(body[0], ast.Return)
                or not isinstance(body[0].value, ast.Dict)
                or any(k.arg is None for k in expression.keywords)
                or any(not isinstance(n, (ast.Name, ast.Constant)) for n in bound.arguments.values())
            ):
                fail(module, expression, "Type member factory must return one fixed dictionary from plain inputs")

            class Bind(ast.NodeTransformer):
                def visit_Name(self, node):
                    if node.id == "self":
                        fail(module, expression, "Type member factory cannot read model state")
                    if node.id in bound.arguments:
                        return copy.deepcopy(bound.arguments[node.id])
                    if node.id not in namespace or method.__globals__.get(node.id) is not namespace[node.id]:
                        fail(module, expression, "Type member factory global requires the same visible definition")
                    return node

                def visit_Call(self, node):
                    if qualified_kind(node, kind) not in ("types.vector", "types.matrix", "types.struct"):
                        fail(module, expression, "Type member factory may only construct builtin numerical types")
                    return self.generic_visit(node)

                def visit_Attribute(self, node):
                    root = node
                    while isinstance(root, ast.Attribute):
                        root = root.value
                    if isinstance(root, ast.Name) and root.id == "self":
                        fail(module, expression, "Type member factory cannot read model state")
                    if not isinstance(root, ast.Name) or root.id in bound.arguments:
                        fail(module, expression, "Type member factory attributes must refer to type modules")
                    return self.generic_visit(node)

            result = Bind().visit(copy.deepcopy(body[0].value))
            # The result grammar is checked by the existing type resolver. Its
            # model tensor axes still come from the original dimension rules.
            cls._package_type_methods = {**getattr(cls, "_package_type_methods", {}), name: method}
            facts.append(
                dict(
                    kind="type_member_factory",
                    definition=method.__module__ + "." + method.__qualname__,
                    line=expression.lineno,
                    sha256=hashlib.sha256(source.encode()).hexdigest(),
                )
            )
            return result

        class Expand(ast.NodeTransformer):
            def visit_Call(self, node):
                node = self.generic_visit(node)
                if qualified_kind(node, kind) != "types.struct":
                    return node
                keywords, seen = [], set()
                for keyword in node.keywords:
                    if keyword.arg is not None:
                        pairs = [(keyword.arg, keyword.value)]
                    else:
                        values = members(keyword.value)
                        if any(not isinstance(k, ast.Constant) or type(k.value) is not str for k in values.keys):
                            fail(module, keyword.value, "Type member factory requires literal member names")
                        pairs = [(k.value, v) for k, v in zip(values.keys, values.values)]
                    for name, value in pairs:
                        if not name.isidentifier() or name in seen:
                            fail(module, node, "Type member expansion requires unique valid member names")
                        seen.add(name)
                        keywords.append(ast.keyword(name, value))
                node.keywords = keywords
                return node

        for function in construction_functions(cls):
            Expand().visit(function)
    return facts
