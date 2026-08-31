import pytest

from taichi_forge.lang import impl
from taichi_forge.lang.exception import TaichiRuntimeError


class _FakeCompletion:
    program_domain = -1

    def __init__(self, sequence, *, fail=False):
        self.sequence = sequence
        self.fail = fail
        self.ready = False
        self.polls = 0

    def done(self):
        self.polls += 1
        if self.fail:
            raise TaichiRuntimeError("completion query failed")
        return self.ready


def test_retain_uses_a_fixed_incremental_poll_budget():
    runtime = impl.PyTaichi()

    for backlog in (1, 8, 64):
        runtime.clear_runtime_submission_owners()
        completions = []
        total_polls = 0
        for sequence in range(backlog):
            completion = _FakeCompletion(sequence)
            completions.append(completion)
            runtime.retain_runtime_submission_owner(completion, object())
            updated_total = sum(item.polls for item in completions)
            assert updated_total - total_polls <= 8
            total_polls = updated_total

        for completion in completions:
            completion.ready = True
        before_collect = sum(item.polls for item in completions)
        runtime.collect_ready_runtime_submission_owners()
        assert sum(item.polls for item in completions) - before_collect == min(
            backlog, 8
        )
        assert len(runtime._runtime_submission_owners) == max(0, backlog - 8)

        before_full_collect = sum(item.polls for item in completions)
        runtime.collect_ready_runtime_submission_owners(_limit=None)
        assert sum(item.polls for item in completions) - before_full_collect == max(
            0, backlog - 8
        )
        assert not runtime._runtime_submission_owners


def test_registry_rejects_aba_and_release_checks_completion_identity():
    runtime = impl.PyTaichi()
    first = _FakeCompletion(1)
    runtime.retain_runtime_submission_owner(first, object())

    reused_key = _FakeCompletion(1)
    with pytest.raises(RuntimeError, match="completion key was reused"):
        runtime.retain_runtime_submission_owner(reused_key, object())

    assert runtime.transfer_runtime_submission_owner(first, object())
    runtime.release_runtime_submission_owner(reused_key)
    key = runtime._runtime_submission_key(first)
    assert runtime._runtime_submission_owners[key][0] is first
    runtime.release_runtime_submission_owner(first)
    assert not runtime._runtime_submission_owners


def test_registry_collects_one_exact_completion_behind_the_poll_cursor():
    runtime = impl.PyTaichi()
    completions = [_FakeCompletion(sequence) for sequence in range(24)]
    for completion in completions:
        runtime.retain_runtime_submission_owner(completion, object())
    for completion in completions:
        completion.polls = 0

    target = completions[-1]
    target.ready = True
    assert runtime.collect_ready_runtime_submission_owner(target)
    assert target.polls == 1
    assert len(runtime._runtime_submission_owners) == len(completions) - 1
    assert all(completion.polls == 0 for completion in completions[:-1])


def test_completion_query_errors_propagate_without_partial_release():
    runtime = impl.PyTaichi()
    ready = _FakeCompletion(1)
    runtime.retain_runtime_submission_owner(ready, object())
    ready.ready = True
    failing = _FakeCompletion(2, fail=True)

    with pytest.raises(TaichiRuntimeError, match="completion query failed"):
        runtime.retain_runtime_submission_owner(failing, object())
    assert len(runtime._runtime_submission_owners) == 2

    with pytest.raises(TaichiRuntimeError, match="completion query failed"):
        runtime.collect_ready_runtime_submission_owners()
    assert len(runtime._runtime_submission_owners) == 2
