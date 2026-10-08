from contextlib import nullcontext

import pytest

import taichi_forge as ti
from taichi_forge.lang.enums import AutodiffMode
from tests import test_utils


@pytest.mark.parametrize("failure", [None, "body", "exit"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_repeated_forward_calls_restore_modes(failure, monkeypatch):
    check_repeated_calls("forward", failure, monkeypatch)


@pytest.mark.parametrize("failure", [None, "body", "exit"])
@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], debug=True, offline_cache=False)
def test_repeated_validation_calls_restore_modes(failure, monkeypatch):
    check_repeated_calls("tape", failure, monkeypatch)


def check_repeated_calls(kind, failure, monkeypatch):
    x = ti.field(ti.f32, shape=(), needs_grad=True, needs_dual=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True, needs_dual=True)
    x[None] = 3

    @ti.kernel
    def add_one():
        loss[None] += x[None]

    @ti.kernel
    def add_two():
        loss[None] += 2 * x[None]

    manager = ti.ad.FwdMode(loss=loss, param=x) if kind == "forward" else ti.ad.Tape(loss, validation=True)

    def fail_on_exit():
        raise RuntimeError("injected exit failure")

    if failure == "exit":
        monkeypatch.setattr(manager, "clear_seed" if kind == "forward" else "grad", fail_on_exit)

    with pytest.raises(RuntimeError, match=f"injected {failure} failure") if failure else nullcontext():
        with manager:
            add_one()
            add_two()
            add_one()
            if failure == "body":
                raise RuntimeError("injected body failure")

    assert add_one._primal.autodiff_mode == AutodiffMode.NONE
    assert add_two._primal.autodiff_mode == AutodiffMode.NONE
    runtime = ti.lang.impl.get_runtime()
    assert runtime.fwd_mode_manager is None
    assert runtime.target_tape is None
    if failure is None:
        assert (loss.dual[None] if kind == "forward" else x.grad[None]) == 4

    # A later ordinary invocation must not execute the forward specialization.
    x.dual[None] = 5
    loss.dual[None] = 0
    add_one()
    assert loss[None] == 15
    assert loss.dual[None] == 0
