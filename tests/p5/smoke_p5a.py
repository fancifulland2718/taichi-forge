"""P5.a smoke test — serial path must still work after adding cache mutex."""
import taichi_forge as ti

ti.init(arch=ti.cpu)


@ti.kernel
def k(n: ti.i32) -> ti.i32:
    s = 0
    for i in range(n):
        s += i
    return s


v1 = k(10)
v2 = k(10)  # hit in-memory cache
v3 = k(100)  # new compile
assert v1 == 45, v1
assert v2 == 45, v2
assert v3 == 4950, v3
print(f"[smoke_p5a] ok: k(10)={v1} k(10)={v2} k(100)={v3}")
