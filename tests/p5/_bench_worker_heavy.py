
import json, os, shutil, sys, time
import taichi as ti

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

SZ = 64
fields = []
kernels = []
for idx in range(n):
    a = float(idx + 1)
    b = float((idx * 7) % 13 + 1)
    c = float((idx * 11) % 19 + 1)

    f = ti.field(ti.f32, shape=SZ)
    fields.append(f)

    @ti.kernel
    def _k(tf: ti.template()):
        for i in tf:
            x = ti.cast(i, ti.f32) * 0.01 + a
            acc = 0.0
            # 3 static-unrolled layers → single offload with big body
            for p in ti.static(range(8)):
                for q in ti.static(range(8)):
                    for r in ti.static(range(4)):
                        t = x + p * 0.11 + q * 0.07 - r * 0.03
                        acc += ti.sin(t * a) * ti.cos(t * b) \
                              + ti.log(1.0 + ti.abs(t * c)) \
                              - ti.exp(-ti.abs(t) * 0.05)
            tf[i] = acc

    _k.__name__ = f"_k_{idx}"
    kernels.append((_k, (f,)))

t0 = time.perf_counter()
if mode == 'parallel':
    from taichi.lang import impl as _impl
    t_mat0 = time.perf_counter()
    specs = []
    for (k, args) in kernels:
        primal = k._primal if hasattr(k, '_primal') and not hasattr(k, 'ensure_compiled') else k
        # Template kernels need args to drive specialization.
        key = primal.ensure_compiled(*args)
        specs.append(primal.compiled_kernels[key])
    t_mat = time.perf_counter() - t_mat0
    prog = _impl.get_runtime().prog
    t_cxx0 = time.perf_counter()
    prog.compile_kernels(prog.config(), specs)
    t_cxx = time.perf_counter() - t_cxx0
    extra = {'t_py_materialize': t_mat, 't_cxx_compile': t_cxx}
else:
    for (k, args) in kernels:
        k(*args)
    extra = {}
elapsed = time.perf_counter() - t0
result = {'elapsed': elapsed}
result.update(extra)
print('__BENCH_RESULT__' + json.dumps(result))
