"""Quick correctness sanity test across CPU / CUDA / Vulkan backends.

Used after each P2.x sub-stage to catch regressions before bench runs.
Exits with non-zero if any backend produces wrong numbers.
"""
import sys
import taichi_forge as ti

EXPECTED_HEAD = [1.5, 2.5, 5.5, 10.5]    # i*i + 1.5 for i=0..3
EXPECTED_TAIL = [3845.5, 3970.5]         # i=62, 63

ARCHES = [("cpu", ti.cpu), ("cuda", ti.cuda), ("vulkan", ti.vulkan)]

failed = []
for name, arch in ARCHES:
    try:
        ti.init(arch=arch, offline_cache=False, log_level="error")
    except Exception as exc:
        print(f"[{name}] init failed: {exc}")
        failed.append(name)
        continue

    @ti.kernel
    def k(x: ti.template()):
        for i in x:
            x[i] = ti.cast(i * i, ti.f32) + 1.5

    a = ti.field(ti.f32, shape=64)
    k(a)
    arr = a.to_numpy().tolist()
    head_ok = arr[:4] == EXPECTED_HEAD
    tail_ok = arr[-2:] == EXPECTED_TAIL
    status = "OK" if (head_ok and tail_ok) else "FAIL"
    print(f"[{name}] {status} head={arr[:4]} tail={arr[-2:]}")
    if not (head_ok and tail_ok):
        failed.append(name)
    ti.reset()

if failed:
    print(f"FAILED backends: {failed}", file=sys.stderr)
    sys.exit(1)
print("All 3 backends produced correct results.")
