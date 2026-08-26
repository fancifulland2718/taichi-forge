"""One private lifecycle boundary for Python-owned CUDA provider calls."""


class _ExternalCudaSubmission:
    def __init__(self, program, resources):
        self._program = program
        self._arrays = tuple(resource.arr for resource in resources)
        self._scope = None
        self._invoked = False

    def __enter__(self):
        if self._scope is not None:
            raise RuntimeError("external CUDA submission scope cannot be re-entered")
        self._scope = self._program._begin_external_cuda_submission()
        return self

    def invoke(self, function, /, *args, **kwargs):
        if self._scope is None:
            raise RuntimeError(
                "external CUDA provider invocation requires an active scope"
            )
        if self._invoked:
            raise RuntimeError(
                "external CUDA submission scope supports one provider call"
            )
        # Mark before crossing the ABI: a provider may enqueue work and then
        # report failure, in which case the resources must still be pinned.
        self._invoked = True
        return function(*args, **kwargs)

    def __exit__(self, exc_type, exc_value, traceback):
        scope = self._scope
        self._scope = None
        if self._invoked:
            try:
                scope._commit(self._arrays, exc_type is not None)
            except Exception:
                if exc_type is None:
                    raise
        return False


def external_cuda_submission(program, resources):
    """Pins resources and records one direct CUDA provider invocation."""

    return _ExternalCudaSubmission(program, tuple(resources))


__all__ = []
