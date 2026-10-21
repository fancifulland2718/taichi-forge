"""Small standalone/borrowed CGraph consumer with no result materialization."""

import array
import importlib.abc
import os
import sys


class NoFrontend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("taichi", "taichi_forge", "numpy"):
            raise AssertionError(fullname)


def reject(exception, action):
    try:
        action()
    except exception:
        return
    raise AssertionError(f"Expected {exception}")


def read(value, code):
    data = array.array(code)
    data.frombytes(value.read())
    return data.tolist()


def main(mode, arch, path, execution):
    if mode == "standalone":
        sys.meta_path.insert(0, NoFrontend())
        from taichi_forge_runtime import aot

        runtime = aot.Runtime(
            arch,
            device_index=(
                int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", 0))
                if arch == "vulkan"
                else 0
            ),
        )
    else:
        import taichi_forge as ti
        from taichi_forge_runtime import aot

        ti.init(arch=getattr(ti, arch), offline_cache=False, cpu_max_num_threads=4)
        runtime = ti.aot.borrow_runtime()
    def bind(module, arguments, gradients):
        if execution == "plan":
            values = dict(arguments, amount=aot.Scalar("i32", arguments["amount"]),
                          dx=gradients["x"], dloss=gradients["loss"])
            bound = module.prepare_execution("step", values)
        else:
            bound = module.graph("step").bind(arguments, gradients=gradients)
            if execution == "prepared":
                bound = bound.prepare()
        if execution != "ordinary":
            # Two advances share the vector frame; read_energy has one scalar.
            assert bound.info["native_resource_bytes"] == 16, bound.info
        return bound

    with runtime:
        module = runtime.load_module(path)
        sibling = runtime.load_module(path)
        state = runtime.ndarray("i32", 2)
        state.write(bytes(8))
        x, dx = runtime.ndarray("f32", 4), runtime.ndarray("f32", 4)
        loss, dloss = runtime.ndarray("f32", 1), runtime.ndarray("f32", 1)
        x.write(array.array("f", [1, 2, -3, 0.5]).tobytes())
        dx.write(bytes(16))
        loss.write(bytes(4))
        dloss.write(array.array("f", [1]).tobytes())
        arguments = dict(state=state, amount=3, x=x, loss=loss)
        gradients = dict(x=dx, loss=dloss)
        graph = module.graph("step")
        incomplete = dict(arguments)
        del incomplete["amount"]
        reject(aot.AotError, lambda: graph(incomplete, gradients=gradients))
        assert read(state, "i") == [0, 0]
        step = bind(module, arguments, gradients)
        for _ in range(4):
            assert step() is None
        assert read(state, "i") == [24, 24]
        assert read(loss, "f") == [57]
        assert read(dx, "f") == [8, 16, -24, 4]
        assert module.kernel("read_energy")(loss, gradients={0: dloss}) == 57
        assert module.kernel("advance")(state, -2) == (22, -2, 2)
        alternate = bind(sibling, dict(arguments, amount=-2), gradients)
        alternate()
        assert read(state, "i") == [18, 18]
        # Close immediately after asynchronous dispatches; storage and code must
        # survive through completion, while the independent module stays valid.
        step()
        module.close()
        reject(RuntimeError, step)
        assert read(state, "i") == [24, 24]
        alternate()
        sibling.close()
        assert read(state, "i") == [20, 20]
        assert read(loss, "f") == [99.75]
        assert read(dx, "f") == [14, 28, -42, 7]
        reject(RuntimeError, alternate)
    if mode == "borrowed":
        late = ti.aot.borrow_runtime()
        late_module = late.load_module(path)
        state = late.ndarray("i32", 2)
        x, dx = late.ndarray("f32", 4), late.ndarray("f32", 4)
        loss, dloss = late.ndarray("f32", 1), late.ndarray("f32", 1)
        for value, size in ((state, 8), (x, 16), (dx, 16), (loss, 4), (dloss, 4)):
            value.write(bytes(size))
        active = bind(late_module,
            dict(state=state, amount=1, x=x, loss=loss),
            dict(x=dx, loss=dloss),
        )
        active()
        ti.reset()
        reject(RuntimeError, active)
        late.close()
    print(mode, arch, execution, "discarded returns passed")


if __name__ == "__main__":
    main(*sys.argv[1:])
