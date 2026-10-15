"""Producer-only lowering of DSL definitions into compiled constructors.

The returned Python module is intermediate build input, not a distribution
artifact. Compile it with the host (for example Cython with binding=True and
annotation_typing=False); ship the extension, never this text or generated C.
"""

import ast
import copy
import symtable


def lower_module(source, module_name):
    tree = ast.parse(source)
    symbols = symtable.symtable(source, module_name, "exec")
    tables = {}

    def collect(table):
        tables[(table.get_type(), table.get_name(), table.get_lineno())] = table
        for child in table.get_children():
            collect(child)

    collect(symbols)
    aliases, direct = set(), {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            aliases.update(n.asname or n.name for n in node.names if n.name in ("taichi", "taichi_forge"))
        elif isinstance(node, ast.ImportFrom) and node.module in ("taichi", "taichi_forge"):
            direct.update((n.asname or n.name, n.name) for n in node.names)
    if any(isinstance(n, ast.Name) and n.id.startswith("_forge_host_") for n in ast.walk(tree)):
        raise ValueError("The compiled-host producer prefix is already used by this module")

    def kind(decorator):
        expr = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name) and expr.value.id in aliases:
            return expr.attr
        if isinstance(expr, ast.Name):
            return direct.get(expr.id)
        return None

    factories, entries = [], []

    def constructor(function, name):
        statements = []

        def encode(value):
            if isinstance(value, ast.AST):
                fields = ast.Tuple(elts=[encode(item) for _, item in ast.iter_fields(value)], ctx=ast.Load())
                position = (
                    (
                        value.lineno,
                        value.col_offset,
                        getattr(value, "end_lineno", None),
                        getattr(value, "end_col_offset", None),
                    )
                    if hasattr(value, "lineno")
                    else ()
                )
                variable = "n" + str(len(statements))
                statements.append(
                    ast.Assign(
                        targets=[ast.Name(id=variable, ctx=ast.Store())],
                        value=ast.Call(
                            func=ast.Name(id="_forge_host_node", ctx=ast.Load()),
                            args=[ast.Constant(type(value).__name__), fields, ast.Constant(position)],
                            keywords=[],
                        ),
                    )
                )
                return ast.Name(id=variable, ctx=ast.Load())
            if isinstance(value, list):
                return ast.List(elts=[encode(item) for item in value], ctx=ast.Load())
            return ast.Constant(value)

        result = encode(ast.Module(body=[function], type_ignores=[]))
        statements.append(ast.Return(value=result))
        return ast.FunctionDef(
            name=name,
            args=ast.arguments(
                posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]
            ),
            body=statements,
            decorator_list=[],
        )

    class Lower(ast.NodeTransformer):
        def __init__(self):
            self.parents = []

        def visit_ClassDef(self, node):
            self.parents.append(node)
            self.generic_visit(node)
            self.parents.pop()
            return node

        def visit_FunctionDef(self, node):
            decorated = [kind(d) for d in node.decorator_list]
            dsl = next((k for k in decorated if k in ("kernel", "func", "real_func", "pyfunc")), None)
            if dsl is None:
                self.parents.append(node)
                self.generic_visit(node)
                self.parents.pop()
                return node
            factory_name = "_forge_host_template_" + str(len(factories))
            original = copy.deepcopy(node)
            original.decorator_list = []
            ast.increment_lineno(original, 1 - node.lineno)
            factories.append(constructor(original, factory_name))
            table = tables[("function", node.name, node.lineno)]
            frees = table.get_frees()
            scope = "{" + ", ".join(repr(n) + ": " + n for n in frees) + "}"
            is_class = bool(self.parents and isinstance(self.parents[-1], ast.ClassDef))
            qualname = ".".join([p.name for p in self.parents] + [node.name])
            location = "<compiled:" + module_name + ":" + qualname + ">"
            binder = ast.parse(
                f"_forge_host_bind({factory_name}, globals_getter=lambda: globals(), "
                f"closure_getter=lambda: {scope}, is_class={is_class!r}, filename={location!r})",
                mode="eval",
            ).body
            annotated = [
                (arg.arg, arg.annotation)
                for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                if arg.annotation is not None
            ]
            if node.returns is not None:
                annotated.append(("return", node.returns))
            binder.keywords.append(
                ast.keyword(
                    arg="annotations",
                    value=ast.Dict(
                        keys=[ast.Constant(name) for name, _ in annotated],
                        values=[copy.deepcopy(annotation) for _, annotation in annotated],
                    ),
                )
            )
            node.decorator_list.append(binder)
            # Keep Python decorator order. Cython treats the literal
            # @staticmethod specially even when other decorators wrap it.
            node.decorator_list = [
                (
                    ast.Name(id="_forge_host_staticmethod", ctx=ast.Load())
                    if isinstance(d, ast.Name) and d.id == "staticmethod"
                    else d
                )
                for d in node.decorator_list
            ]
            if dsl != "pyfunc":
                node.body = [
                    ast.Raise(
                        exc=ast.Call(
                            func=ast.Name(id="RuntimeError", ctx=ast.Load()),
                            args=[ast.Constant("Compiled DSL definitions require their Forge decorator")],
                            keywords=[],
                        )
                    )
                ]
            entries.append({"name": qualname, "kind": dsl, "closure": list(frees), "is_class": is_class})
            return node

    tree = Lower().visit(tree)
    if not entries:
        return source, ()
    helpers = ast.parse(
        "from builtins import staticmethod as _forge_host_staticmethod\n"
        "from taichi_forge.lang._compiled_template import bind as _forge_host_bind, node as _forge_host_node\n"
    ).body
    insert = 0
    for node in tree.body:
        if (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        ) or (isinstance(node, ast.ImportFrom) and node.module == "__future__"):
            insert += 1
        else:
            break
    tree.body[insert:insert] = helpers + factories
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n", tuple(entries)
