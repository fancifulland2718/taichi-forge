"""Device completion handles, independent of the DSL and result consumption."""

import threading


class SubmissionTicket:
    """Completion returned by ``PreparedCall.submit()``.

    ``done()`` polls; ``wait()`` waits without taking the runtime submission
    lock. Neither freezes output arrays nor returns a workspace/result lease.
    Explicit owner retirement drains native handles before device destruction,
    so a retained ticket can still report completion or its first error.
    ``close()`` discards this handle without waiting for the submitted work.
    """

    def __init__(self, api, handle, owners):
        if not handle:
            raise RuntimeError("Submission did not return a completion handle")
        self._api = api
        self._handle = handle
        self._owners = owners
        self._completed = False
        self._lock = threading.RLock()

    def _check(self):
        if self._handle is None:
            raise RuntimeError("Submission ticket is closed")

    def done(self):
        with self._lock:
            self._check()
            if not self._completed:
                self._completed = bool(
                    self._api.call(self._api.ti_query_submission_ext, self._handle)
                )
            return self._completed

    def wait(self):
        with self._lock:
            self._check()
            if not self._completed:
                self._api.call(self._api.ti_wait_submission_ext, self._handle)
                self._completed = True
            # Keep owners until explicit close. Dropping their final references
            # here could run resource destructors under the submission lock,
            # turning a completion-only wait into an unrelated retirement wait.

    def close(self):
        with self._lock:
            if self._handle is not None:
                self._api.call(self._api.ti_destroy_submission_ext, self._handle)
                self._handle = None
                self._owners = ()

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
