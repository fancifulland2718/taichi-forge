"""Expose a uniform numeric input when its Host owner optionally allocates the Field."""

import ast


def normalize_storage_choices(constructor, declared_scopes, fail):
    """Both arms name the same managed member; only the allocating arm owns storage."""
    occupied = {n.attr for n in ast.walk(constructor) if isinstance(n, ast.Attribute)}

    class StorageChoices(ast.NodeTransformer):
        def visit_If(self, node):
            self.generic_visit(node)
            for owned, supplied in ((node.body, node.orelse), (node.orelse, node.body)):
                if len(owned) != 1 or len(supplied) != 1:
                    continue
                allocation, statement = owned[0], supplied[0]
                if not isinstance(allocation, ast.Assign) or len(allocation.targets) != 1:
                    continue
                target, factory = allocation.targets[0], allocation.value
                if not (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in declared_scopes(allocation)
                    and isinstance(factory, ast.Call)
                    and isinstance(factory.func, ast.Attribute)
                    and isinstance(factory.func.value, ast.Name)
                    and factory.func.value.id == target.value.id
                    and factory.func.attr in ("field", "vector_field", "matrix_field")
                ):
                    continue
                call = statement.value if isinstance(statement, ast.Expr) else None
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "borrow"
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == target.value.id
                    and len(call.args) == 2
                    and isinstance(call.args[0], ast.Constant)
                    and call.args[0].value == target.attr
                ):
                    continue
                keywords = {kw.arg: kw.value for kw in call.keywords}
                if not {"dtype", "shape"} <= keywords.keys():
                    continue
                # Move the source expression into its original arm. This also preserves
                # the single evaluation of the condition and any input property reads.
                name = target.attr
                private = f"_forge_package_owned_{name}"
                local = f"_forge_package_source_{name}"
                if private in occupied or any(
                    isinstance(part, (ast.Name, ast.arg)) and getattr(part, "id", getattr(part, "arg", None)) == local
                    for part in ast.walk(constructor)
                ):
                    fail(node, "Conditional Field collides with a reserved package storage name")
                occupied.add(private)
                target.attr = private
                target._package_storage_choice = True
                owned.append(
                    ast.copy_location(
                        ast.Assign(
                            [ast.Name(local, ast.Store())],
                            ast.Attribute(ast.Name("self", ast.Load()), private, ast.Load()),
                        ),
                        allocation,
                    )
                )
                supplied[:] = [ast.copy_location(ast.Assign([ast.Name(local, ast.Store())], call.args[1]), statement)]
                call.args[1] = ast.copy_location(ast.Name(local, ast.Load()), call.args[1])
                # Keep original call location for normal Host projection. Make the typed
                # port available unconditionally after the storage decision.
                return [node, statement]
            return node

    ast.fix_missing_locations(StorageChoices().visit(constructor))
