"""Process-isolated lifetime regression, with real AOT AD and sparse callbacks."""

import array
import ctypes
import faulthandler
import sys

import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge_runtime.aot import Runtime


def read(value, dtype="f"):
    result = array.array(dtype)
    result.frombytes(value.read())
    return result.tolist()


class Consumer:
    def __init__(self, artifact):
        self.runtime = Runtime("cpu")
        self.module = self.runtime.load_module(artifact)
        self.x, self.y, self.dx, self.dy = (self.runtime.ndarray("f32", 3) for _ in range(4))
        self.x.write(array.array("f", [1, 3, 5]))
        self.dy.write(array.array("f", [1, 1, 1]))
        self.out = self.runtime.ndarray("i32", 1)
        gradients = {0: self.dx, 1: self.dy}
        self.forward = self.module.kernel("forward").prepare(self.x, self.y, gradients=gradients)
        self.backward = self.module.kernel("backward").prepare(self.x, self.y, gradients=gradients)
        self.count = 0

    def check(self, index):
        self.dx.write(array.array("f", [0, 0, 0]))
        # Reverse mode consumes output adjoints; seed each independent pass.
        self.dy.write(array.array("f", [1, 1, 1]))
        self.forward()
        self.backward()
        self.module.kernel("touch")(index, self.out)
        self.runtime.wait()
        self.count += 1
        assert read(self.x) == [1, 3, 5]
        assert read(self.y) == [3, 15, 35]
        assert read(self.dx) == [4, 8, 12]
        assert read(self.out, "i") == [self.count]


def main(reset_arch, artifact):
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetErrorMode(0x8002)
    faulthandler.enable()
    survivor = Consumer(artifact)
    survivor.check(0)
    for iteration in range(2):
        ti.init(arch=getattr(ti, reset_arch), offline_cache=False, cpu_max_num_threads=2)
        jit = ti.ndarray(ti.i32, 3)
        jit.fill(7)
        ti.sync()
        program = ti.lang.impl.get_runtime().prog
        stats = dict(core.get_host_memory_pool_stats(program))
        other = Consumer(artifact)
        other.check(0)
        survivor.check(2 + 2 * iteration)
        # Creating and executing independent AOT owners must not contribute to
        # the current JIT executor's host allocation statistics.
        assert dict(core.get_host_memory_pool_stats(program)) == stats
        borrower = ti.aot.borrow_runtime()
        borrowed = borrower.ndarray("i32", 3)
        borrowed.write(array.array("i", [11, 13, 17]))
        ti.reset()
        assert borrower._borrower.closed
        survivor.check(6)
        other.check(2)
        other.runtime.close()
        survivor.check(4)

    # Reverse destruction order: closing standalone AOT must preserve JIT.
    ti.init(arch=getattr(ti, reset_arch), offline_cache=False, cpu_max_num_threads=2)
    jit = ti.ndarray(ti.i32, 3)
    jit.fill(19)
    survivor.runtime.close()
    survivor.runtime.close()
    assert jit.to_numpy().tolist() == [19, 19, 19]
    jit.fill(23)
    assert jit.to_numpy().tolist() == [23, 23, 23]
    ti.reset()
    print("CPU AOT owner isolation passed", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
