"""P3.a — unrolling_hard_limit per-loop enforcement.

A single ti.static loop exceeding unrolling_hard_limit must abort with a
TaichiCompilationError early (before fully expanding the body N times).
"""
import time
import taichi_forge as ti
from taichi_forge.lang.exception import TaichiCompilationError


def main():
    ti.init(arch=ti.cpu, offline_cache=False, unrolling_hard_limit=16,
            log_level=ti.WARN)

    N = 8
    x = ti.field(ti.f32, shape=N)

    @ti.kernel
    def run():
        for i in x:
            s = ti.f32(0.0)
            for k in ti.static(range(100)):  # exceeds hard_limit=16
                s += float(k) * ti.cast(i, ti.f32)
            x[i] = s

    t0 = time.perf_counter()
    try:
        run()
    except TaichiCompilationError as e:
        dt = time.perf_counter() - t0
        msg = str(e)
        assert "unrolling_hard_limit=16" in msg, msg
        # Early abort: should be <1s even on a cold start. Body expansion
        # of 100 iters on CPU usually takes significantly longer if not cut.
        print(f"P3.a per-loop OK — aborted in {dt*1000:.1f} ms with message:\n  {msg.splitlines()[0]}")
        return
    raise AssertionError("expected TaichiCompilationError for runaway static loop")


if __name__ == "__main__":
    main()
