"""Consume selected native WHILE without importing the DSL or source kernels."""
import sys
import numpy as np
from taichi_forge_runtime.aot import Runtime

artifact, mode = sys.argv[1:]
with Runtime("cuda") as runtime:
    module = runtime.load_module(artifact)

    def make_arrays():
        values = dict(limits=runtime.ndarray("i32", (3,)), predicate=runtime.ndarray("i32", (1,)),
                      counter=runtime.ndarray("i32", (1,)))
        if mode == "step":
            values.update(out=runtime.ndarray("i32", (66,)), scale=3)
        else:
            values.update({name: runtime.ndarray("f32", (65,)) for name in ("x", "y", "dx", "dy")})
        return values

    def reset(values, n, target, lane=0):
        values["limits"].write(np.array([n, target, -99], dtype=np.int32).tobytes())
        values["predicate"].write(np.array([-99], dtype=np.int32).tobytes())
        if mode == "step":
            values["out"].write(np.full(66, -1, dtype=np.int32).tobytes())
        else:
            values["x"].write((np.arange(65, dtype=np.float32) + lane + 1).tobytes())
            for name, value in (("y", -1), ("dx", 0), ("dy", 1)):
                values[name].write(np.full(65, value, dtype=np.float32).tobytes())

    def check(values, n, target, lane=0):
        iterations = min(target, 4)
        assert np.frombuffer(values["limits"].read(), dtype=np.int32)[2] == iterations
        assert np.frombuffer(values["counter"].read(), dtype=np.int32)[0] == iterations
        if mode == "step":
            expected = np.full(66, -1, dtype=np.int32)
            expected[:n] = 1
            for k in range(1, iterations + 1):
                expected[:max(0, n - 17 * k)] += 1
            expected[65] = 1000 + 3 * iterations
            np.testing.assert_array_equal(np.frombuffer(values["out"].read(), dtype=np.int32), expected)
        else:
            expected_y = np.full(65, -1, dtype=np.float32)
            expected_dx = np.zeros(65, dtype=np.float32)
            for k in range(iterations):
                size = max(0, n - 17 * k)
                x = np.arange(size, dtype=np.float32) + lane + 1
                expected_y[:size] = x * x
                expected_dx[:size] += 2 * x
            np.testing.assert_array_equal(np.frombuffer(values["y"].read(), dtype=np.float32), expected_y)
            np.testing.assert_array_equal(np.frombuffer(values["dx"].read(), dtype=np.float32), expected_dx)

    values = make_arrays()
    call = module.prepare_execution(mode, values)
    for n, target in ((65, 2), (33, 99), (0, 3), (65, 0)):
        reset(values, n, target)
        call()
        check(values, n, target)
    print("ORDINARY-WHILE-WORKSPACE", call.info["workspace_bytes"])
    call.close()
    domains = [runtime.execution_domain(instance_capacity=1) for _ in range(2)]
    owners = [make_arrays() for _ in domains]
    calls = [module.prepare_execution(mode, owner, execution_domain=domain)
             for domain, owner in zip(domains, owners)]
    for call in calls:
        call.initialize()
    print("INDEPENDENT-WHILE-WORKSPACE", [call.info["workspace_bytes"] for call in calls])
    for targets in (((65, 2), (33, 99)), ((0, 3), (65, 0))):
        for lane, (n, target) in enumerate(targets):
            reset(owners[lane], n, target, lane)
        runtime.wait()
        completions = [call.submit() for call in calls]
        for completion in completions:
            completion.wait()
            completion.close()
        for lane, (n, target) in enumerate(targets):
            check(owners[lane], n, target, lane)
    calls[0].close()
    domains[0].close()
    reset(owners[1], 65, 3, 1)
    runtime.wait()
    completion = calls[1].submit()
    completion.wait()
    completion.close()
    check(owners[1], 65, 3, 1)
    calls[1].close()
    domains[1].close()
    module.close()
assert "taichi_forge" not in sys.modules
print("CUDA-RANGE-WHILE", mode, "runtime-only checks passed")
