"""Per-run subprocess helper for run_bench.py.

Reads _HK_NAME, _HK_LLVM_OPT, _HK_SPV_OPT from the environment,
initialises Taichi with the requested options, compiles the named
hard kernel, runs it once, then prints elapsed_ms to stdout.

Must be a separate file so each invocation gets a fresh Python interpreter
with no leftover LLVM JIT state or Taichi program singletons.
"""
import os
import sys
import time

# suppress Taichi's kernel-IR printing by forcing the env knob OFF before import
os.environ.setdefault("TI_PRINT_IR", "0")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

kernel_name = os.environ.get("_HK_NAME", "")
llvm_opt    = int(os.environ.get("_HK_LLVM_OPT", "3"))
spv_opt     = int(os.environ.get("_HK_SPV_OPT",  "3"))

if not kernel_name:
    print("_HK_NAME not set", file=sys.stderr)
    sys.exit(1)

import taichi as ti  # noqa: E402

# Initialise with the requested optimisation levels; no offline cache → cold compile.
# external_optimization_level is only relevant for GPU/Vulkan (SPIR-V) paths.
# For CPU benchmarks we only vary llvm_opt_level.
ti.init(
    arch=ti.cpu,
    offline_cache=False,
    log_level="error",
    print_ir=False,
    llvm_opt_level=llvm_opt,
)

from hard_kernels import HARD_KERNELS  # noqa: E402

factory = next((f for n, f in HARD_KERNELS if n == kernel_name), None)
if factory is None:
    print(f"Unknown kernel: {kernel_name!r}", file=sys.stderr)
    sys.exit(1)

# Build the kernel (field allocation).
kernel_fn = factory(ti)

# Time the first call (triggers JIT compile + one execution).
t0 = time.perf_counter()
kernel_fn()
ti.sync()
t1 = time.perf_counter()

print(f"{(t1 - t0) * 1000.0:.3f}")
