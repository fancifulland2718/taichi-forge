"""Consume export.py's module using only the runtime wheel and Python stdlib."""

import argparse
import array

from taichi_forge_runtime import aot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arch", choices=("cpu", "cuda", "vulkan"))
    parser.add_argument("module")
    parser.add_argument("--library", help="Optional path to the shared Forge native library")
    parser.add_argument("--runtime", help="Optional directory containing LLVM runtime bitcode")
    parser.add_argument("--prepared", action="store_true", help="Prepare fixed native execution before running")
    options = parser.parse_args()
    with aot.Runtime(options.arch, library_path=options.library, runtime_path=options.runtime) as runtime:
        module = runtime.load_module(options.module)
        x, y, dx, dy = [runtime.ndarray("f32", 4) for _ in range(4)]
        x.write(array.array("f", [1, 2, 3, 4]))
        dx.write(array.array("f", [0] * 4))
        dy.write(array.array("f", [1] * 4))
        call = module.graph("differentiate").bind({"x": x, "y": y}, gradients={"x": dx, "y": dy})
        if options.prepared:
            call = call.prepare()
        call()
        result, gradient = array.array("f"), array.array("f")
        result.frombytes(y.read())
        gradient.frombytes(dx.read())
        assert result.tolist() == [3, 8, 15, 24]
        assert gradient.tolist() == [4, 6, 8, 10]
        print("y:", result.tolist(), "dx:", gradient.tolist())
        if options.prepared:
            print("prepared:", call.info)
            call.close()


if __name__ == "__main__":
    main()
