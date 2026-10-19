"""Complete prepared workspaces with separate completion and consumption."""

from collections import deque
from dataclasses import dataclass
import operator
import threading
from types import MappingProxyType


class WorkspaceBusyError(RuntimeError):
    """No workspace is free; outstanding results still belong to their leases."""


@dataclass
class _Slot:
    call: object
    arrays: object
    generation: int = 0
    state: str = "free"
    ticket: object = None


class WorkspacePool:
    """One module's code, independently prepared private resources per slot.

    Create through ``Module.prepare_workspace_pool``. Arrays are allocated once
    per slot; shared arguments retain their identity. All submissions use the
    runtime's existing ordered queue. A completed result remains leased until
    release/discard; this API does not introduce concurrent device streams.
    """

    def __init__(
        self, module, plan, capacity, *, arguments=None, arrays=None, gradients=None
    ):
        from ._plan import ExecutionPlan

        if isinstance(capacity, bool):
            raise TypeError("Workspace capacity must be a positive integer")
        capacity = operator.index(capacity)
        if capacity <= 0:
            raise ValueError("Workspace capacity must be positive")
        shared, descriptions = dict(arguments or {}), dict(arrays or {})
        if shared.keys() & descriptions.keys():
            raise ValueError(
                "Shared arguments and workspace arrays must have different names"
            )
        if isinstance(plan, str):
            plan = module.execution_plan(plan)
        if not isinstance(plan, ExecutionPlan):
            raise TypeError("Expected an ExecutionPlan or an exported plan name")
        # Freeze the caller's descriptions before allocating any slot.
        descriptions = {name: dict(spec) for name, spec in descriptions.items()}
        gradient_names = dict(gradients or {})
        self._module = module
        self._lock = threading.RLock()
        self._reclaim_lock = threading.Lock()
        self._retirement_lock = threading.RLock()
        self._slots = []
        self._free, self._discarded = deque(), deque()
        self._closed = self._retired = False
        self._failure = None
        try:
            for index in range(capacity):
                owned = {}
                call = None
                try:
                    for name, spec in descriptions.items():
                        owned[name] = module._runtime.ndarray(**spec)
                    bindings = dict(shared, **owned)
                    bound_gradients = {
                        name: bindings[value] for name, value in gradient_names.items()
                    }
                    # Each prepare rebuilds the full private resource DAG. A
                    # binding-family sibling would incorrectly share state.
                    call = module.prepare_execution(
                        plan, bindings, gradients=bound_gradients
                    )
                    self._slots.append(_Slot(call, MappingProxyType(owned)))
                    self._free.append(index)
                except BaseException:
                    if call is not None:
                        try:
                            call.close()
                        except Exception:
                            pass
                    for array in owned.values():
                        try:
                            array.close()
                        except Exception:
                            pass
                    raise
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def _admit(self):
        if self._closed:
            raise RuntimeError("Workspace pool is closed")
        if self._failure is not None:
            raise RuntimeError(
                "Workspace pool failed; close it before preparing a replacement"
            ) from self._failure

    def _record_failure(self, failure):
        with self._lock:
            if self._failure is None:
                self._failure = failure

    def acquire(self, *, wait=False):
        """Lease a free slot, optionally waiting for an already discarded result.

        Waiting never takes a still-owned result away from its consumer. If all
        slots have live leases, both modes raise WorkspaceBusyError.
        """
        with self._lock:
            self._admit()
            if self._free:
                return self._lease_free()
        if not self._reclaim_lock.acquire(blocking=wait):
            raise WorkspaceBusyError(
                "Another caller is reclaiming a discarded workspace"
            )
        try:
            return self._acquire_discarded(wait)
        finally:
            self._reclaim_lock.release()

    def _lease_free(self):
        index = self._free.popleft()
        slot = self._slots[index]
        slot.generation += 1
        slot.state = "leased"
        return WorkspaceLease(self, index, slot.generation)

    def _acquire_discarded(self, wait):
        while True:
            with self._lock:
                self._admit()
                if self._free:
                    return self._lease_free()
                if not self._discarded:
                    raise WorkspaceBusyError(
                        "All workspace results are leased; release or discard a lease"
                    )
                index = self._discarded[0]
                slot = self._slots[index]
                ticket = slot.ticket
            # No pool or runtime submission lock during device waits/queries.
            try:
                if wait:
                    ticket.wait()
                elif not ticket.done():
                    raise WorkspaceBusyError(
                        "Discarded workspace is still in flight; acquire(wait=True) may wait"
                    )
            except WorkspaceBusyError:
                raise
            except BaseException as error:
                self._record_failure(error)
                raise
            with self._lock:
                self._admit()
                if (
                    self._discarded
                    and self._discarded[0] == index
                    and slot.ticket is ticket
                ):
                    try:
                        ticket.close()
                    except BaseException as error:
                        self._record_failure(error)
                        raise
                    self._discarded.popleft()
                    slot.ticket, slot.state = None, "free"
                    self._free.append(index)

    def initialize(self):
        """Initialize private resources at a cold boundary before leasing slots."""
        with self._lock:
            self._admit()
            if len(self._free) != len(self._slots):
                raise WorkspaceBusyError(
                    "Initialize workspaces before acquiring leases"
                )
            try:
                for slot in self._slots:
                    slot.call.initialize()
            except BaseException as error:
                self._record_failure(error)
                raise

    @property
    def info(self):
        """Cold diagnostics; no native counters are queried on warm acquisition."""
        with self._lock:
            self._admit()
            calls = tuple(slot.call.info for slot in self._slots)
            return dict(
                capacity=len(self._slots),
                available=len(self._free),
                discarded=len(self._discarded),
                array_bytes=sum(
                    array.nbytes
                    for slot in self._slots
                    for array in slot.arrays.values()
                ),
                argument_bytes=sum(call["argument_bytes"] for call in calls),
                workspace_bytes=sum(call["workspace_bytes"] for call in calls),
                resource_bytes=sum(call["resource_bytes"] for call in calls),
                native_resource_bytes=sum(
                    call["native_resource_bytes"] for call in calls
                ),
            )

    def close(self):
        with self._retirement_lock:
            self._close()

    def _close(self):
        with self._lock:
            if self._retired:
                return
            self._closed = True
            slots = tuple(self._slots)
        failure = None
        for slot in slots:
            if slot.ticket is not None:
                try:
                    slot.ticket.wait()
                except BaseException as error:
                    if failure is None:
                        failure = error
                try:
                    slot.ticket.close()
                    slot.ticket = None
                except BaseException as error:
                    if failure is None:
                        failure = error
            actions = (
                slot.call.close,
                *(array.close for array in slot.arrays.values()),
            )
            for action in actions:
                try:
                    action()
                except BaseException as error:
                    if failure is None:
                        failure = error
        with self._lock:
            self._retired = failure is None
        if failure is not None:
            self._record_failure(failure)
            raise failure

    def __enter__(self):
        self._admit()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class WorkspaceLease:
    """One acquisition, one submission, and explicit ownership of its results.

    ``release()`` waits and returns the slot; ``discard()`` relinquishes results
    immediately but reuse still waits for completion. Array references are valid
    only while this lease is live. Keeping a raw reference after release does
    not reserve its storage. Leaving a context calls release().
    """

    def __init__(self, pool, index, generation):
        self._pool, self._index, self._generation = pool, index, generation
        self._released = False
        self._lock = threading.RLock()

    def _slot(self):
        # The caller holds pool._lock. Lease identity prevents stale reuse.
        if self._released or self._pool._closed:
            raise RuntimeError("Workspace lease is released or its pool is closed")
        slot = self._pool._slots[self._index]
        if slot.generation != self._generation:
            raise RuntimeError("Workspace lease generation is retired")
        return slot

    @property
    def arrays(self):
        with self._pool._lock:
            return self._slot().arrays

    @property
    def index(self):
        """Stable slot ordinal, valid only during this acquisition."""
        with self._pool._lock:
            self._slot()
            return self._index

    def submit(self):
        with self._lock, self._pool._lock:
            self._pool._admit()
            slot = self._slot()
            if slot.state != "leased":
                raise RuntimeError("A workspace lease can submit only once")
            try:
                slot.ticket = slot.call.submit()
                slot.state = "submitted"
            except BaseException as error:
                slot.state = "failed"
                self._pool._record_failure(error)
                raise
        return self

    def _completion(self):
        with self._pool._lock:
            slot = self._slot()
            if slot.ticket is None:
                raise RuntimeError("Workspace lease has no submitted completion")
            return slot.ticket

    def done(self):
        with self._lock:
            ticket = self._completion()
            try:
                return ticket.done()
            except BaseException as error:
                self._pool._record_failure(error)
                raise

    def wait(self):
        with self._lock:
            ticket = self._completion()
            try:
                ticket.wait()
            except BaseException as error:
                self._pool._record_failure(error)
                raise

    def _return(self, discard):
        with self._lock:
            if self._released:
                return
            pool = self._pool
            with pool._lock:
                if pool._closed:
                    self._released = True
                    return
                slot = self._slot()
                if slot.state == "failed":
                    self._released = True
                    return
                ticket = slot.ticket
            if ticket is not None and not discard:
                try:
                    ticket.wait()
                except BaseException as error:
                    pool._record_failure(error)
                    raise
            with pool._lock:
                self._released = True
                if pool._closed:
                    return
                if ticket is not None and discard:
                    slot.state = "discarded"
                    pool._discarded.append(self._index)
                else:
                    if ticket is not None:
                        try:
                            ticket.close()
                        except BaseException as error:
                            slot.state = "failed"
                            pool._record_failure(error)
                            raise
                    slot.ticket, slot.state = None, "free"
                    pool._free.append(self._index)

    def release(self):
        self._return(False)

    def discard(self):
        self._return(True)

    close = release

    def __enter__(self):
        with self._pool._lock:
            self._slot()
        return self

    def __exit__(self, *_):
        self.release()

    def __del__(self):
        try:
            self.discard()
        except Exception:
            pass
