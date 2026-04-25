"""P2.c tier spot: one subprocess run per tier; mat14 cold compile."""
import os, sys, time, shutil, subprocess, json


def child():
    tier = sys.argv[2]
    arch = sys.argv[3]
    cache = os.path.expanduser("~/.cache/taichi")
    if os.path.exists(cache):
        shutil.rmtree(cache, ignore_errors=True)
    import taichi as ti
    ti.init(arch=getattr(ti, arch), offline_cache=False, compile_tier=tier)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from heavy_kernels import make_mat14
    k = make_mat14(ti)
    t0 = time.perf_counter()
    k()
    ti.sync()
    dt = time.perf_counter() - t0
    sys.stdout.write(f"__RESULT__{json.dumps({'tier': tier, 'seconds': dt})}\n")
    sys.stdout.flush()


def parent():
    arch = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    tiers = ["fast", "balanced", "full"]
    results = {}
    print(f"[P2.c tier spot] arch={arch}", flush=True)
    for t in tiers:
        r = subprocess.run(
            [sys.executable, __file__, "--child", t, arch],
            capture_output=True, text=True, timeout=400,
        )
        if r.returncode != 0:
            print(f"  tier={t:<8s}  FAILED rc={r.returncode}")
            print("  stderr:", r.stderr[-500:])
            continue
        for line in r.stdout.splitlines():
            if line.startswith("__RESULT__"):
                data = json.loads(line[len("__RESULT__"):])
                results[t] = data["seconds"]
                print(f"  tier={t:<8s}  cold_compile={data['seconds']:.3f}s",
                      flush=True)
                break
    b = results.get("balanced", 0.0)
    print("\nSummary (seconds, single cold run; mat14):")
    for t in tiers:
        s = results.get(t, 0.0)
        delta = (s - b) / b * 100 if b else 0.0
        print(f"  {t:<8s}  {s:.3f}s   ({delta:+.1f}% vs balanced)")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--child":
        child()
    else:
        parent()
