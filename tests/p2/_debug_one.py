import os, shutil, sys, time
tier = sys.argv[1] if len(sys.argv) > 1 else "fast"
kernel_name = sys.argv[2] if len(sys.argv) > 2 else "sph_force"
home = os.path.expanduser("~")
p = os.path.join(home, ".cache", "taichi", "ticache")
if os.path.isdir(p):
    shutil.rmtree(p, ignore_errors=True)
sys.path.insert(0, r"d:\taichi\tests\p2")
import taichi_forge as ti
import heavy_kernels as hk
print(f"[child] tier={tier} kernel={kernel_name}", flush=True)
ti.init(arch=ti.vulkan, offline_cache=False, compile_tier=tier,
        random_seed=42, default_fp=ti.f32, advanced_optimization=True,
        log_level=ti.WARN)
factory = getattr(hk, "make_" + kernel_name)
print(f"[child] factory ready, building fields...", flush=True)
run = factory(ti)
print(f"[child] starting cold compile + first launch", flush=True)
t0 = time.perf_counter(); run(); ti.sync()
print(f"[child] cold= {time.perf_counter()-t0:.3f}s", flush=True)
t1 = time.perf_counter(); run(); ti.sync()
print(f"[child] warm= {time.perf_counter()-t1:.3f}s", flush=True)
