"""Cold consumers of the compiler's final range-controller qualification."""


def device_range_dispatch_rejection(task):
    """Return a diagnostic, never infer a proof from a shape or TLS byte count.

    The version is emitted from RangeForBounds::supports_device_dispatch().
    Host empty-group elision has a separate, intentionally broader contract.
    """
    if task.backend not in ("cuda", "vulkan"):
        return "backend has no device range controller"
    if task.task_type != "range_for":
        return "not a range task"
    if task.device_range_dispatch_version != 1:
        bounds = task.range_bounds
        proof = "unknown" if bounds is None else bounds.empty_group_elision
        return f"no compiled device range controller proof ({proof})"
    return None
