"""Capture qualified sort definitions into the existing deployment lowering.

Only declaration identities cross this boundary. Neither producer arrays nor
their workspace are retained by the exported component.
"""

import json
from types import SimpleNamespace

from taichi_forge.graph._ir import GraphAccess, NativeCallNode, ResourceEffect, RuntimeBinding
from taichi_forge.graph._native import NativeGraphExecutable
from taichi_forge.lang.exception import TaichiCompilationError, TaichiRuntimeError


def capture_sequence(sequence, capture):
    from ._algorithms import PrimitiveSequence
    from ._sequence_export import _sort

    if type(sequence) is not PrimitiveSequence or sequence._fused_plan is not None:
        raise TaichiCompilationError(
            "Component native capture requires an unfused qualified primitive sequence"
        )
    calls = tuple(sequence._calls)
    if not calls or any(call.kind != "sort" for call in calls):
        raise TaichiCompilationError(
            "Component native capture currently qualifies single-key native sort only"
        )
    operations = []
    context = SimpleNamespace(
        retained_arrays=SimpleNamespace(require=capture.native_array), operations=operations
    )
    for call in calls:
        # Reuse the ordinary exporter: exact ordering/NaN policy, native route,
        # packed storage, dtype, workspace and selected strategy validation.
        _sort(call, context)
    return _ComponentSortDefinition(operations)


class _ComponentSortDefinition(NativeGraphExecutable):
    def __init__(self, operations):
        from taichi_forge.aot._component import _digest

        self._operations = json.dumps(operations, sort_keys=True)
        self.semantic_key = _digest(operations)
        self._names = tuple(
            sorted({value for op in operations for key, value in op.items() if key != "kind"})
        )

    @property
    def operations(self):
        return json.loads(self._operations)

    @property
    def runtime_arg_schema(self):
        return tuple(RuntimeBinding(name, "ndarray") for name in self._names)

    @property
    def resource_effects(self):
        return tuple(ResourceEffect(name, GraphAccess.READ_WRITE) for name in self._names)

    @property
    def graph_ir_node(self):
        return NativeCallNode(
            name="component_sort_" + self.semantic_key,
            bindings=self.runtime_arg_schema,
            effects=self.resource_effects,
            opaque=False,
            semantic_fingerprint=self.semantic_key,
        )

    def _export_aot(self, context):
        # Sort length and scratch are resolved by existing Runtime preparation
        # from this invocation's arrays, never from the producer's shape.
        for operation in self.operations:
            context.operations.append(
                {
                    key: value if key == "kind" else context.binding(value, resource=True)
                    for key, value in operation.items()
                }
            )

    def run(self, runtime_args=None):
        raise TaichiRuntimeError(
            "A component native definition must execute through its exported instance plan"
        )
