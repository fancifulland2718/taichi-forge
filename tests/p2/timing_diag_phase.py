"""阶段补充测试：测量优化后的 sph_force / mat14 在 CPU 与 Vulkan 后端的冷编译时间。

通过 --arch 选项切换 CPU / Vulkan 后端，逐个 kernel 测量首次调用 (含编译) 的总耗时。
所有 kernel 均关闭 offline_cache，确保测的是冷编译。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
os.environ["TI_PRINT_IR"] = "0"

import taichi_forge as ti  # noqa: E402
from heavy_kernels import make_sph_force, make_mat14  # noqa: E402


def measure(name, factory):
    fn = factory(ti)
    t0 = time.perf_counter()
    fn()
    ti.sync()
    elapsed = time.perf_counter() - t0
    print(f"[COLD] {name} first-call TOTAL: {elapsed:.3f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["cpu", "vulkan"], required=True)
    args = parser.parse_args()

    arch = ti.cpu if args.arch == "cpu" else ti.vulkan
    ti.init(arch=arch, offline_cache=False, print_ir=False)
    print(f"[init] arch={ti.cfg.arch}", flush=True)

    measure("mat14", make_mat14)
    # sph_force 单独 init 一次以避免共用 program 状态影响
    ti.reset()
    ti.init(arch=arch, offline_cache=False, print_ir=False)
    measure("sph_force", make_sph_force)


if __name__ == "__main__":
    main()
