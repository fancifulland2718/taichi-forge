"""P3 — early-abort latency benchmark.

Compares compile time of a runaway `ti.static(range(N))` loop
  (1) with the hard-limit DISABLED (baseline: unrolling runs to N)
  (2) with a tight unrolling_hard_limit (aborts almost immediately)

The speed-up ratio quantifies the user-visible benefit: instead of waiting
for tens of seconds on a pathological static loop, they get a clear error
in a few milliseconds.
"""
import time
import taichi_forge as ti
from taichi_forge.lang.exception import TaichiCompilationError


SIZES = [400, 800, 1600]


def build_and_time(N, hard_limit):
    ti.reset()
    ti.init(arch=ti.cpu, offline_cache=False,
            unrolling_hard_limit=hard_limit,
            log_level=ti.ERROR)
    x = ti.field(ti.f32, shape=4)

    @ti.kernel
    def run():
        for i in x:
            s = ti.f32(0.0)
            for k in ti.static(range(N)):
                s += float(k) * ti.cast(i, ti.f32)
            x[i] = s

    t0 = time.perf_counter()
    status = "ok"
    try:
        run()
        ti.sync()
    except TaichiCompilationError:
        status = "aborted"
    return time.perf_counter() - t0, status


def main():
    print(f"{'N':>6} {'baseline (s)':>14} {'abort (s)':>12} {'speedup':>10}")
    for N in SIZES:
        t_base, s_base = build_and_time(N, hard_limit=0)
        t_abort, s_abort = build_and_time(N, hard_limit=50)
        ratio = t_base / t_abort if t_abort > 0 else float("inf")
        print(f"{N:>6} {t_base:>14.3f} {t_abort:>12.4f} {ratio:>10.1f}x   baseline={s_base}  abort={s_abort}")


if __name__ == "__main__":
    main()
