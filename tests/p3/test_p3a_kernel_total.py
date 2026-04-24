"""P3.a — unrolling_kernel_hard_limit cumulative enforcement.

Two modest nested ti.static loops (8 x 8 = 64 iters) pass per-loop limit=16
but exceed the kernel-total cap=32. Must abort.
"""
import taichi as ti
from taichi.lang.exception import TaichiCompilationError


def main():
    ti.init(arch=ti.cpu, offline_cache=False,
            unrolling_hard_limit=100,  # per-loop cap high enough
            unrolling_kernel_hard_limit=32,
            log_level=ti.WARN)

    N = 4
    x = ti.field(ti.f32, shape=N)

    @ti.kernel
    def run():
        for i in x:
            s = ti.f32(0.0)
            for a in ti.static(range(8)):
                for b in ti.static(range(8)):
                    s += float(a * b)
            x[i] = s

    try:
        run()
    except TaichiCompilationError as e:
        msg = str(e)
        assert "unrolling_kernel_hard_limit=32" in msg, msg
        print(f"P3.a kernel-total OK — message: {msg.splitlines()[0]}")
        return
    raise AssertionError("expected TaichiCompilationError for nested static overflow")


if __name__ == "__main__":
    main()
