"""P3.b — func_inline_depth_limit enforcement.

Three nested @ti.func calls (depth 3) must be rejected when the limit is
set to 2.
"""
import taichi as ti
from taichi.lang.exception import TaichiCompilationError


def main():
    ti.init(arch=ti.cpu, offline_cache=False,
            func_inline_depth_limit=2,
            log_level=ti.WARN)

    @ti.func
    def level3(x: ti.f32) -> ti.f32:
        return x * 2.0

    @ti.func
    def level2(x: ti.f32) -> ti.f32:
        return level3(x) + 1.0

    @ti.func
    def level1(x: ti.f32) -> ti.f32:
        return level2(x) * 3.0

    out = ti.field(ti.f32, shape=4)

    @ti.kernel
    def run():
        for i in out:
            out[i] = level1(ti.cast(i, ti.f32))

    try:
        run()
    except TaichiCompilationError as e:
        msg = str(e)
        assert "func_inline_depth_limit=2" in msg, msg
        print(f"P3.b depth-limit OK — message: {msg.splitlines()[0]}")
        return
    raise AssertionError("expected TaichiCompilationError for deep func chain")


if __name__ == "__main__":
    main()
