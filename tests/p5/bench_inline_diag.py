"""Single-process CPU heavy bench — no subprocess wrapper."""
import time, sys, os
sys.stdout.reconfigure(line_buffering=True)

out = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "inline_result.txt"), "w")
def say(m):
    print(m, flush=True); out.write(m + "\n"); out.flush()

import taichi as ti
say(f"taichi={ti.__file__}")
say(f"has compile_kernels on ti: {'compile_kernels' in dir(ti)}")

ti.init(arch=ti.cpu, offline_cache=False, num_compile_threads=8, log_level=ti.WARN)

SZ = 64
N = 12
fields, kernels = [], []
for idx in range(N):
    a = float(idx + 1); b = float((idx*7)%13+1); c = float((idx*11)%19+1)
    f = ti.field(ti.f32, shape=SZ); fields.append(f)
    @ti.kernel
    def _k(tf: ti.template()):
        for i in tf:
            x = ti.cast(i, ti.f32) * 0.01 + a
            acc = 0.0
            for p in ti.static(range(8)):
                for q in ti.static(range(8)):
                    for r in ti.static(range(4)):
                        t = x + p*0.11 + q*0.07 - r*0.03
                        acc += ti.sin(t*a)*ti.cos(t*b) + ti.log(1.0+ti.abs(t*c)) - ti.exp(-ti.abs(t)*0.05)
            tf[i] = acc
    _k.__name__ = f"_k_{idx}"
    kernels.append((_k, f))

# Serial: cold-compile one at a time
t0 = time.perf_counter()
for k, f in kernels:
    k(f)
serial = time.perf_counter() - t0
say(f"serial compile+launch: {serial*1000:.1f} ms")

# Reset & reinit for parallel path
ti.reset()
ti.init(arch=ti.cpu, offline_cache=False, num_compile_threads=8, log_level=ti.WARN)

fields, kernels = [], []
for idx in range(N):
    a = float(idx + 1); b = float((idx*7)%13+1); c = float((idx*11)%19+1)
    f = ti.field(ti.f32, shape=SZ); fields.append(f)
    @ti.kernel
    def _k(tf: ti.template()):
        for i in tf:
            x = ti.cast(i, ti.f32) * 0.01 + a
            acc = 0.0
            for p in ti.static(range(8)):
                for q in ti.static(range(8)):
                    for r in ti.static(range(4)):
                        t = x + p*0.11 + q*0.07 - r*0.03
                        acc += ti.sin(t*a)*ti.cos(t*b) + ti.log(1.0+ti.abs(t*c)) - ti.exp(-ti.abs(t)*0.05)
            tf[i] = acc
    _k.__name__ = f"_k_p_{idx}"
    kernels.append((_k, f))

from taichi.lang import impl as _impl
t_m0 = time.perf_counter()
specs = []
for k, f in kernels:
    primal = k._primal if hasattr(k, '_primal') and not hasattr(k, 'ensure_compiled') else k
    say(f"kernel {k.__name__}: type={type(primal).__name__}, has ensure_compiled={hasattr(primal, 'ensure_compiled')}")
    key = primal.ensure_compiled(f)
    say(f"  key={key}, has compiled_kernels={hasattr(primal, 'compiled_kernels')}")
    specs.append(primal.compiled_kernels[key])
    break
say(f"specs count={len(specs)}")
t_mat = time.perf_counter() - t_m0
say(f"materialize 1 kernel took {t_mat*1000:.1f} ms")

out.close()
