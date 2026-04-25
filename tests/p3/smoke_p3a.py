"""P3.a — smoke test: a plain ti.static(range(N)) loop unrolled below
unrolling_hard_limit must compile and run correctly (semantics unchanged)."""
import taichi_forge as ti


def main():
    ti.init(arch=ti.cpu, offline_cache=False, unrolling_hard_limit=128,
            log_level=ti.WARN)

    N = 16
    x = ti.field(ti.f32, shape=N)

    @ti.kernel
    def run():
        for i in x:
            s = ti.f32(0.0)
            for k in ti.static(range(8)):
                s += float(k) * ti.cast(i, ti.f32)
            x[i] = s

    run()
    ti.sync()
    arr = x.to_numpy()
    # sum_{k=0..7} k*i = 28*i
    for i in range(N):
        assert abs(arr[i] - 28.0 * i) < 1e-5, (i, arr[i])
    print("P3.a smoke OK — static(range(8)) under hard_limit=128 compiled and ran correctly.")


if __name__ == "__main__":
    main()
