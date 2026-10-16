"""Exercise each CUB native port/type contract through fixed AOT capture."""

import array
import json
import os
import sys
import zipfile

from taichi_forge_runtime.aot import Runtime
from tests.python.aot_binding_constraints_probe import failure


def main(mode, artifact):
    if mode == "borrowed":
        import taichi_forge as ti

        ti.init(arch=ti.cuda, offline_cache=False)
        runtime = ti.aot.borrow_runtime()
    else:
        runtime = Runtime("cuda")
    with runtime:
        runtime.configure_provider(
            "cub_source",
            adapter_path=os.environ["TI_CUB_SOURCE_TEST_ADAPTER"],
            library_path=os.environ["TI_CUB_SOURCE_TEST_LIBRARY"],
        )
        module = runtime.load_module(artifact)
        with zipfile.ZipFile(artifact) as archive:
            descriptions = [
                json.loads(archive.read(name))
                for name in archive.namelist()
                if name.startswith("executions/") and name.endswith(".json")
            ]
        for document in descriptions:
            parameters = document["resources"][0]["parameters"]
            operation, n = parameters["operation"], parameters["num_items"]
            keys = [(i * 7) % 11 + (2**33 if "pairs_u64" in operation else 0) for i in range(max(1, n))]
            data = dict(input0=keys, output0=[73] * max(1, n))
            if operation.startswith("radix_"):
                data.update(input1=list(range(max(1, n))), output1=[73] * max(1, n))
            elif operation.startswith("select_"):
                data.update(input1=[i % 2 for i in range(max(1, n))], output1=[73])
            elif operation.startswith("segmented_"):
                data["input1"] = [1 | (1 << 5) | (1 << 31), 0] if n else [1]
            arrays = {}
            for port, values in data.items():
                dtype = "u64" if "pairs_u64" in operation and port in ("input0", "output0") else "u32"
                arrays[port] = runtime.ndarray(dtype, len(values))
                arrays[port].write(array.array("Q" if dtype == "u64" else "I", values))

            def read(port):
                result = array.array("Q" if arrays[port].dtype == "u64" else "I")
                result.frombytes(arrays[port].read())
                return list(result)

            with module.prepare_execution(f"{operation}_{n}", arrays) as call:
                assert read("output0") == [73] * max(1, n)
                assert call.info["execution_kind"] == "cuda_graph"
                for _ in range(2):
                    call()
                    if not n:
                        assert read("output0") == [73]
                        if operation.startswith("select_"):
                            assert read("output1") == [0]
                    elif operation.startswith("radix_"):
                        order = sorted(range(n), key=lambda i: keys[i])
                        assert read("output0") == [keys[i] for i in order]
                        assert read("output1") == order
                    elif operation.startswith("select_"):
                        reference = [value for i, value in enumerate(keys) if i % 2]
                        assert read("output1") == [len(reference)]
                        assert read("output0")[: len(reference)] == reference
                    else:
                        reference, total = [], 0
                        for i, value in enumerate(keys):
                            if operation.startswith("segmented_") and i in (0, 5, 31):
                                total = 0
                            reference.append(total + value if "inclusive" in operation else total)
                            total += value
                        assert read("output0") == reference
                wrong = dict(arrays, input0=runtime.ndarray("f32", max(1, n)))
                failure(lambda: module.prepare_execution(f"{operation}_{n}", wrong), "dtype")
            if not n:
                empty = dict(arrays, input0=runtime.ndarray("u32", 0), output0=runtime.ndarray("u32", 0))
                if operation.startswith("select_"):
                    empty["input1"] = runtime.ndarray("u32", 0)
                with module.prepare_execution(f"{operation}_{n}", empty) as call:
                    call()
                    runtime.wait()
    if mode == "standalone":
        assert "taichi_forge" not in sys.modules


if __name__ == "__main__":
    main(*sys.argv[1:])
