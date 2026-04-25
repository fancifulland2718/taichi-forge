
import json, os, shutil, sys, time
import taichi_forge as ti

mode   = sys.argv[1]
arch   = sys.argv[2]
n      = int(sys.argv[3])
thread = int(sys.argv[4])
cache_dirs = sys.argv[5].split('|') if len(sys.argv) > 5 else []

for d in cache_dirs:
    if d and os.path.isdir(d):
        try: shutil.rmtree(d)
        except Exception: pass

arch_obj = {'cpu': ti.cpu, 'cuda': ti.cuda, 'vulkan': ti.vulkan}[arch]
ti.init(arch=arch_obj, offline_cache=False,
        num_compile_threads=thread, log_level=ti.WARN)

kernels = []
for idx in range(n):
    a = float(idx + 1)
    b = float((idx * 7) % 13 + 1)

    @ti.kernel
    def _k() -> ti.f32:
        s = 0.0
        x = 0.5
        # Deliberately unrolled big loop → heavier LLVM optimization.
        for i in range(256):
            s += ti.sin(x * a + i * 0.1) * ti.cos(x * b - i * 0.07) \
                 + ti.log(1.0 + ti.abs(x + i * 0.01)) * a \
                 - ti.exp(-ti.abs(x + i * 0.02) * 0.01) * b
        return s

    _k.__name__ = f"_k_{idx}"
    kernels.append(_k)

t0 = time.perf_counter()
if mode == 'parallel':
    # Phase split: how much is Python-side materialize vs. C++ parallel compile?
    from taichi_forge.lang import impl as _impl
    t_mat0 = time.perf_counter()
    specs = []
    for k in kernels:
        primal = k._primal if hasattr(k, '_primal') and not hasattr(k, 'ensure_compiled') else k
        key = primal.ensure_compiled()
        specs.append(primal.compiled_kernels[key])
    t_mat = time.perf_counter() - t_mat0
    prog = _impl.get_runtime().prog
    t_cxx0 = time.perf_counter()
    prog.compile_kernels(prog.config(), specs)
    t_cxx = time.perf_counter() - t_cxx0
    extra = {'t_py_materialize': t_mat, 't_cxx_compile': t_cxx}
else:
    for k in kernels:
        k()
    extra = {}
elapsed = time.perf_counter() - t0
result = {'elapsed': elapsed}
result.update(extra)
print('__BENCH_RESULT__' + json.dumps(result))
