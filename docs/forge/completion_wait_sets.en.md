# Completion sets

[中文](completion_wait_sets.zh.md)

`CompletionSet` waits for any member of a frozen collection of exact submission
prefixes. The same implementation accepts ordinary JIT Graph tickets, ordinary
AOT `SubmissionTicket`, and independent-domain `Completion`. CPU, CUDA and Vulkan
use their existing completion primitives. CUDA and qualified fixed-compute Vulkan
domains both provide these exact tickets; see [domain boundaries](cuda_execution_domains.en.md).

Check `runtime_contract_manifest()["features"].get("completion_wait_any", False)`
before using this addition. Runtime-only consumers import it without the DSL:

```python
from taichi_forge_runtime.aot import CompletionSet

with CompletionSet([input_done, render_done], poll_interval=0.001) as progress:
    result = progress.wait(timeout=0.02)
    if result.status == "ready":
        consume_completed_prefix(result.index)
    elif result.status == "timeout":
        do_other_host_work()
```

The frontend exposes the same class as `ti.graph.CompletionSet`.
`result.index` is the first ready member's original input index. It is `None`
for `timeout` and `cancelled`. A failed backend completion raises its error;
it is never converted to readiness, timeout or a zero GPU duration.

## Lifetime and cancellation

- Construction snapshots native completion state. Original host ticket handles
  can close afterwards; later submissions cannot extend these exact prefixes.
- `wait(timeout=None)` waits indefinitely. Zero makes one nonblocking scan.
  A set can be queried repeatedly; readiness does not consume a member.
- `cancel()` is terminal and wakes all waiting callers. It does not cancel GPU
  work or release an application's result/workspace lease.
- `close()` cancels and releases the set. An already executing native wait keeps
  its own reference. A new wait racing close may report a closed handle.
- Waiting and cancellation do not acquire runtime submission locks. Neither
  operation initializes resources, re-prepares Graphs or submits device work.
- JIT tickets preserve their existing pacer and completion-observation handling
  when a ready member is reported. Native snapshots hold completion ownership,
  not a new copy of application outputs.

All members must come from the same loaded native runtime library; they may
belong to different runtime instances and supported backends. Closing an owner
still follows its existing retirement contract. A retained set can observe
completed or failed tokens afterwards.

## Observation policy

This is polling-based wait-any, not an operating-system notification for each
GPU event. Only the calling thread scans members and sleeps between scans; no
thread per ticket, background service, CUDA host callback or Graph node is added.
The member count has no arbitrary cap; the public C ABI uses a uint64 count,
subject to addressable host storage.

`poll_interval` is configurable in seconds, with a 1 ms default. Smaller values
trade more backend queries and CPU wakeups for shorter observation delay. The
interval is not a real-time deadline: scheduling and backend queries add delay.
Windows uses a high-resolution waitable timer and a cancellation event without
changing global timer resolution. Older systems lacking this timer use their
ordinary timer granularity. Other platforms use a condition-variable wait;
Linux qualification is deferred for this release work.

A frozen set can be reused for the same prefixes. Construct a new set for new
submissions; this allocates host bookkeeping, not another prepared GPU instance.
The waiter creates one timer when it first needs to sleep on Windows, regardless
of the number of tickets. Immediate readiness and zero-timeout scans need no
waiter timer.
