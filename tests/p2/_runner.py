"""Per-run subprocess helper for run_vs_baseline.py.

Environment variables consumed
-------------------------------
  _HK_NAME     : kernel name from HEAVY_KERNELS registry
  _HK_ARCH     : taichi arch string  (cpu | cuda)
  _HK_TIMEOUT  : per-kernel timeout override in seconds (default 600)

Prints the elapsed compile+first-run milliseconds as the LAST line of
stdout, and exits 0 on success or non-zero on failure.

Must be a separate file so each invocation begins with a fresh Python
interpreter (no leftover LLVM JIT state or Taichi program singletons).
"""
import os
import sys
import time

# Suppress Taichi's kernel-IR printing before importing anything taichi-related.
os.environ["TI_PRINT_IR"] = "0"

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

kernel_name = os.environ.get("_HK_NAME", "")
arch_str    = os.environ.get("_HK_ARCH", "cpu").lower()

if not kernel_name:
    print("_HK_NAME not set", file=sys.stderr)
    sys.exit(1)

import taichi as ti  # noqa: E402

arch_map = {
    "cpu":    ti.cpu,
    "cuda":   ti.cuda,
    "vulkan": ti.vulkan,
    "metal":  ti.metal,
    "opengl": ti.opengl,
}
arch = arch_map.get(arch_str, ti.cpu)

# Initialise with no offline cache (pure cold compile).
try:
    ti.init(
        arch=arch,
        offline_cache=False,
        log_level="error",
        print_ir=False,
    )
except Exception as e:
    print(f"ti.init failed: {e}", file=sys.stderr)
    sys.exit(2)

# Verify we actually got the requested arch (Taichi may silently fall back).
# ti.cfg.arch returns "arch.x64", "arch.cuda", "arch.vulkan", etc.
actual_arch = str(ti.cfg.arch).lower()  # e.g. "arch.x64"
# Build an alias table so "cpu" matches "x64" and "x86_64"
_arch_aliases = {
    "cpu": ("x64", "x86_64", "cpu", "arm64"),
    "cuda": ("cuda",),
    "vulkan": ("vulkan",),
    "metal": ("metal",),
    "opengl": ("opengl",),
}
expected_tokens = _arch_aliases.get(arch_str, (arch_str,))
if not any(tok in actual_arch for tok in expected_tokens):
    print(f"ARCH_FALLBACK: requested={arch_str}, got={actual_arch}", file=sys.stderr)
    sys.exit(3)

from heavy_kernels import HEAVY_KERNELS  # noqa: E402

factory = next((f for n, f, *_ in HEAVY_KERNELS if n == kernel_name), None)
if factory is None:
    print(f"Unknown kernel: {kernel_name!r}", file=sys.stderr)
    sys.exit(1)

# Build fields + kernel object.
try:
    kernel_fn = factory(ti)
except Exception as e:
    print(f"Factory error: {e}", file=sys.stderr)
    sys.exit(4)

# Time: first call triggers JIT compile + one execution.
t0 = time.perf_counter()
try:
    kernel_fn()
    ti.sync()
except Exception as e:
    print(f"Kernel run error: {e}", file=sys.stderr)
    sys.exit(5)
t1 = time.perf_counter()

# Always emit ms as the last line (parents parse this via reversed-line scan).
print(f"{(t1 - t0) * 1000.0:.1f}")
