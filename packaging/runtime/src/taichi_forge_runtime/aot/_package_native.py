"""Cold provider bindings for package Graphs; execution stays in ExecutionPlan."""

from dataclasses import dataclass
from types import SimpleNamespace

from ._native import native_references
from ._runtime import NativeResource


class NativeObject:
    """Typed Host facade over an existing Runtime-owned native resource."""

    def __init__(self, package, resource):
        if not isinstance(resource, NativeResource) or resource._runtime is not package.runtime:
            raise ValueError("Native object requires this package's Runtime resource")
        self.package, self.resource = package, resource
        from ._package import _construction
        construction = _construction.get()
        if construction is not None and construction[0] is package:
            construction[3].callback(self.close)

    def _check(self):
        self.package._check()
        self.resource._check()

    def close(self):
        self.resource.close()

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.close()


class NativeRecording:
    """One provider operation with explicit port types and effects.

    Symbols describe consumer bindings, never pointers or producer snapshots.
    Direct calls and enclosing Graphs use the same native preparation path.
    """

    def __init__(self, package, provider, operation, *, version=1, parameters=None,
                 bindings, schemas, effects):
        from ._plan import ExecutionPlan
        description = dict(kind="native", provider=provider, operation=operation,
                           version=version, parameters=dict(parameters or {}),
                           bindings={name: name for name in bindings})
        native_references(description)
        if bindings.keys() != schemas.keys() or bindings.keys() != effects.keys():
            raise ValueError("Native ports require complete type and effect contracts")
        if any(effect not in ("read", "write", "read_write") for effect in effects.values()):
            raise ValueError("Unsupported native resource effect")
        # Validate and snapshot through the existing portable schema.
        import json
        self.description = json.loads(ExecutionPlan([description]).to_json())["operations"][0]
        self.package = package
        self.bindings, self.schemas, self.effects = dict(bindings), dict(schemas), dict(effects)
        self._graph = None

    def execute(self, arguments):
        if self._graph is None:
            builder = self.package.graph_builder()
            builder.append_native(self, admission="auto")
            self._graph = builder.compile()
        self._graph.run(arguments)

    def close(self):
        if self._graph is not None:
            self._graph.close()
            self._graph = None


@dataclass(frozen=True)
class _Selection:
    definition: object
    manifest: object


class NativeGraphDefinition:
    """Frozen package Plan using its established native executor.

    Package preparation already retains immutable bindings. This implements
    the construction protocol, without importing the frontend recipe search.
    """

    def __init__(self, graph):
        self.graph = graph

    def select_execution_recipe(self, *, queue="preserve", binding_reuse="require"):
        if queue != "preserve" or binding_reuse not in ("require", "prefer"):
            raise ValueError("Package materialization currently requires the preserved execution queue")
        return _Selection(self, SimpleNamespace(is_baseline=True))

    def materialize(self, selection):
        if not isinstance(selection, _Selection) or selection.definition is not self:
            raise ValueError("Execution selection belongs to another definition")
        from ._package_graph import Graph
        source = self.graph
        source.package._check()
        return _Materialized(Graph(source.package, source.plan, dict(source.arguments),
            dict(source.captures), list(source.admissions), list(source.range_checks), dict(source.plan_values),
            native_effects=dict(source.native_effects), native_extents=dict(source.native_extents),
            workspace_lanes=source._workspace_lanes, workspace_saturation=source._workspace_saturation))


class _Materialized:
    def __init__(self, executor):
        self.executor = executor

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.executor.close()
