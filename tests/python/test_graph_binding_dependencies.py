"""Derived resources must declare a frozen, acyclic symbolic parent closure."""

import json

import pytest
import taichi_forge as ti
from taichi_forge.graph._ir import RuntimeBinding
from taichi_forge.graph._native import DispatchGraphAction, NativeGraphExecutable, native_action_manifest


class DerivedExecutable(NativeGraphExecutable):
    recordable_action = None
    runtime_arg_schema = (RuntimeBinding("x", "ndarray"),)
    derived_runtime_arg_schema = tuple(RuntimeBinding(name, "ndarray") for name in ("a", "b"))

    def __init__(self, dependencies):
        self.recordable_action = DispatchGraphAction(((object(), ()),), binding_dependencies=dependencies)


def test_graph_dependency_manifest_freezes_input_and_preserves_legacy_schema():
    dependencies = {"a": ["x"], "b": ["a", "x"]}
    executable = DerivedExecutable(dependencies)
    dependencies["a"].append("b")
    manifest = native_action_manifest(executable)
    assert manifest.schema_version == 5
    assert manifest.binding_dependencies == (("a", ("x",)), ("b", ("a", "x")))
    assert json.loads(json.dumps(manifest.to_dict()))["binding_dependencies"] == [["a", ["x"]], ["b", ["a", "x"]]]
    with pytest.raises(TypeError):
        executable.recordable_action.binding_dependencies["a"] = ("b",)
    legacy = native_action_manifest(DerivedExecutable({}))
    assert legacy.schema_version == 4
    assert "binding_dependencies" not in legacy.to_dict()


@pytest.mark.parametrize(
    "dependencies, message",
    [
        ({"undeclared": ["x"]}, "declared derived"),
        ({"a": ["missing"]}, "undeclared parent"),
        ({"a": []}, "undeclared parent"),
        ({"a": [None]}, "undeclared parent"),
        ({"a": ["a"]}, "cycle"),
        ({"a": ["b"], "b": ["a"]}, "cycle"),
    ],
)
def test_graph_dependency_manifest_rejects_invalid_closures(dependencies, message):
    with pytest.raises(ti.TaichiRuntimeError, match=message):
        native_action_manifest(DerivedExecutable(dependencies))


def test_custom_graph_action_dependency_declaration_is_checked_and_sampled_once():
    class CustomAction(DispatchGraphAction):
        dependencies = {"a": "x"}

        @property
        def binding_dependencies(self):
            return self.dependencies

    class CustomExecutable(DerivedExecutable):
        recordable_action = CustomAction(((object(), ()),))

        def __init__(self):
            pass

    executable = CustomExecutable()
    with pytest.raises(TypeError, match="sequences of names"):
        native_action_manifest(executable)
    executable.recordable_action.dependencies = {"a": iter(("x",))}
    assert native_action_manifest(executable).binding_dependencies == (("a", ("x",)),)
