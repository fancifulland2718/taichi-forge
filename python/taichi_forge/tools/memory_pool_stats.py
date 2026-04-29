"""R1.c: ``ti.tools.memory_pool_stats()`` read-only diagnostic API.

Returns a snapshot of the host- and device-side memory pool counters
maintained by the C++ runtime. The snapshot is taken under each pool's
own mutex so the values are internally consistent; calls add only one
extra mutex acquire each and are safe to call at any time.

Use this primarily to understand allocator behavior under a workload —
how often kernels hit the device free-list cache, how much memory is
parked in the cache vs. live on device, and how the host-side bump
allocator is growing across compiles.

Example::

    import taichi_forge as ti
    ti.init(arch=ti.cpu)
    # ... run your workload ...
    snap = ti.tools.memory_pool_stats()
    print(snap["host"]["allocate_count"], snap["host"]["raw_bytes"])
    print(snap["device"]["cache_hit_count"], snap["device"]["cache_miss_count"])
"""
from __future__ import annotations

from typing import Any, Dict

from taichi_forge._lib import core as _ti_core


def memory_pool_stats() -> Dict[str, Dict[str, int]]:
    """Return a snapshot of host + device memory pool counters.

    Returns
    -------
    dict
        ``{"host": {...}, "device": {...}}``. The ``host`` dict has keys
        ``allocate_count``, ``release_count``, ``bytes_allocated_total``,
        ``bytes_released_total``, ``raw_chunks``, ``raw_bytes``,
        ``unified_chunks``. The ``device`` dict additionally has
        ``cache_hit_count``, ``cache_miss_count``, ``cached_blocks``,
        ``cached_bytes``.

        On non-LLVM-device builds the ``device`` dict is still returned
        but most counters will be zero (the device pool only sees traffic
        on CUDA/AMDGPU backends).

    Notes
    -----
    Counters are cumulative since process start. To measure a region,
    snapshot before and after and diff fields manually.
    """
    return {
        "host": dict(_ti_core.get_host_memory_pool_stats()),
        "device": dict(_ti_core.get_device_memory_pool_stats()),
    }


__all__ = ["memory_pool_stats"]
