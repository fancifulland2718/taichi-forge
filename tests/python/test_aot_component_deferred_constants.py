"""Deferred Final annotations preserve guards without executing Python hints."""
from __future__ import annotations

from typing import Final
import typing as hints

import numpy as np
import pytest
import taichi_forge as ti
from taichi_forge.aot import Component
from taichi_forge.aot._component import Array
from tests import test_utils


VALUE: Final[int] = 3
OFFSET: hints.Final[int] = 7
UNUSED: Final[int] = 99


@ti.func
def _value(i):
    return VALUE + OFFSET + i


@ti.data_oriented
class _State:
    def __init__(self, n):
        self.output = ti.ndarray(ti.i32, n)

    @ti.kernel
    def step(self):
        for i in range(self.output.shape[0]):
            self.output[i] = _value(i)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_deferred_final_guards_values_and_import_bindings(monkeypatch):
    first, second = _State(3), _State(17)
    environment = Component(output=Array(ti.i32, 1))
    code = environment.compile(first.step)
    assert environment.compile(second.step) is code
    code.run(environment, second)
    np.testing.assert_array_equal(second.output.to_numpy(), np.arange(17) + 10)
    monkeypatch.setitem(globals(), "UNUSED", 100)
    assert environment.compile(second.step) is code
    monkeypatch.setitem(globals(), "VALUE", 5)
    changed = environment.compile(second.step)
    assert changed.semantic_key != code.semantic_key
    changed.run(environment, second)
    np.testing.assert_array_equal(second.output.to_numpy(), np.arange(17) + 12)
    monkeypatch.setitem(globals(), "Final", int)
    with pytest.raises(ti.TaichiCompilationError, match="undeclared global/closure.*VALUE"):
        environment.compile(second.step)
    monkeypatch.setitem(globals(), "Final", hints.Final)
    assert environment.compile(second.step) is changed
    monkeypatch.setitem(globals(), "hints", object())
    with pytest.raises(ti.TaichiCompilationError, match="undeclared global/closure.*OFFSET"):
        environment.compile(second.step)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_deferred_final_never_evaluates_annotation_expressions(monkeypatch):
    state = _State(1)
    environment = Component(output=Array(ti.i32, 1))
    annotations = globals()["__annotations__"]
    calls = []

    class Trap:
        @property
        def Final(self):
            calls.append("getter")
            return Final

    def trap():
        calls.append("call")
        return int

    monkeypatch.setitem(globals(), "trap", trap)
    monkeypatch.setitem(globals(), "proxy", Trap())
    for annotation in ("trap()", "proxy.Final[int]", "Final.__getitem__(int)", "Final["):
        monkeypatch.setitem(annotations, "VALUE", annotation)
        with pytest.raises(ti.TaichiCompilationError, match="undeclared global/closure.*VALUE"):
            environment.compile(state.step)
    # The Final role, not the (unevaluated) type parameter, authorizes capture.
    monkeypatch.setitem(annotations, "VALUE", "Final[trap()]")
    code = environment.compile(state.step)
    code.run(environment, state)
    assert state.output[0] == 10
    assert calls == []
