"""Cold definition acquisition shared by source and compiled host callables.

Only syntax and diagnostic locations are cached. Scope values are read when a
specialization needs them; mutable AST nodes never escape into the template.
This is a producer-side boundary, independent of AOT packaging and executors.
"""

import ast
from dataclasses import dataclass
import inspect
import textwrap
from typing import Optional

from taichi_forge.lang._compile_profile import python_compile_profile_event
from taichi_forge.lang._wrap_inspect import getsourcefile, getsourcelines
from taichi_forge.lang.ast._ast_template import ASTTemplate


@dataclass(frozen=True)
class DefinitionTemplate:
    template: Optional[ASTTemplate]
    filename: str
    lines: list
    start_lineno: int
    source_info_cache: Optional[dict]


class CallableDefinition:
    def __init__(self, function, *, is_class):
        self.function = function
        self.compiled = getattr(function, "_taichi_compiled_template", None)
        self.is_class = self.compiled.is_class if self.compiled is not None else is_class
        self.signature = self.compiled.signature if self.compiled is not None else inspect.signature(function)
        self.cached_template = None

    def scope(self):
        if self.compiled is not None:
            return self.compiled.scope()
        result = self.function.__globals__.copy()
        if self.function.__closure__:
            result.update(
                (name, cell.cell_contents)
                for name, cell in zip(self.function.__code__.co_freevars, self.function.__closure__)
            )
        return result

    def acquire(self, *, cache):
        """Return fresh syntax and its common diagnostic/template metadata.

        Both adapters publish only a complete result. With caching disabled,
        use the freshly parsed/constructed tree directly, without cloning or
        retaining it. Profiling keeps the existing source/parse stage names.
        """
        prefix = f"python.frontend.{self.function.__name__}"
        definition = self.cached_template if cache else None
        if self.compiled is not None:
            with python_compile_profile_event(f"{prefix}.compiled_template"):
                if definition is None:
                    tree, lines = self.compiled.instantiate(self.function.__name__)
                    definition = self._result(tree, self.compiled.filename, lines, 1, cache)
                else:
                    tree = definition.template.instantiate()
        else:
            with python_compile_profile_event(f"{prefix}.source"):
                if definition is None:
                    filename = getsourcefile(self.function)
                    lines, start = getsourcelines(self.function)
                    lines = [textwrap.fill(line, tabsize=4, width=9999) for line in lines]
                    source = textwrap.dedent("\n".join(lines))
            with python_compile_profile_event(f"{prefix}.ast_parse"):
                if definition is None:
                    tree = ast.parse(source)
                    tree.body[0].decorator_list = []
                    definition = self._result(tree, filename, lines, start, cache)
                else:
                    tree = definition.template.instantiate()
        if cache:
            self.cached_template = definition
        return tree, definition

    @staticmethod
    def _result(tree, filename, lines, start, cache):
        return DefinitionTemplate(ASTTemplate(tree) if cache else None, filename, lines, start, {} if cache else None)
