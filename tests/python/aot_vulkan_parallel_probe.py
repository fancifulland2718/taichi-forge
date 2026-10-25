"""Stdlib-only parallel consumer, also qualified with a borrowed Program."""

import array
import json
import os
import sys


class NoFrontend:
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in ("taichi", "taichi_forge", "numpy"):
            raise AssertionError("Source-free parallel imported " + fullname)


def read(value, code="f"):
    result = array.array(code)
    result.frombytes(value.read())
    return result.tolist()


def check(runtime, artifact, borrowed):
    from taichi_forge_runtime.aot import AotError, Scalar

    module = runtime.load_module(artifact)
    domains = [runtime.execution_domain(capacity=2, instance_capacity=1, gpu_timing=True) for _ in range(2)]
    frames = []
    calls = []
    for instance, domain in enumerate(domains):
        arrays = {f"{name}{lane}": runtime.ndarray("f32", 17 + lane * 14)
                  for lane in range(2) for name in ("x", "y", "dx", "dy")}
        arrays["out"] = runtime.ndarray("u64", 3)
        args = dict(arrays, epoch=Scalar("u64", 0))
        gradients = {f"{name}{lane}": arrays[f"d{name}{lane}"] for lane in range(2) for name in ("x", "y")}
        if instance == 0:
            try:
                module.prepare_execution("step", args, gradients=gradients)
            except AotError as error:
                assert "independent Vulkan" in str(error)
            else:
                raise AssertionError("Vulkan parallel accepted an ordinary queue")
            try:
                module.prepare_execution("step", dict(args, y1=arrays["y0"]), gradients=gradients,
                                         execution_domain=domain, mutable_parameters=("epoch",))
            except (AotError, ValueError) as error:
                assert "overlap" in str(error).lower() or "disjoint" in str(error).lower(), error
            else:
                raise AssertionError("Aliasing branches accepted")
        call = module.prepare_execution("step", args, gradients=gradients,
                                        execution_domain=domain, mutable_parameters=("epoch",))
        assert call.info["execution_kind"] == "vulkan_commands", call.info
        print("PARALLEL_INSTANCE_INFO=" + json.dumps(dict(call.info), sort_keys=True))
        call.initialize()
        calls.append(call)
        frames.append(arrays)

    kept = []
    for turn in range(3):
        tickets = []
        expected = []
        for instance, (call, arrays) in enumerate(zip(calls, frames)):
            host = [[float(i + 1 + instance + turn) for i in range(17 + lane * 14)] for lane in range(2)]
            for lane in range(2):
                arrays[f"x{lane}"].write(array.array("f", host[lane]))
            epoch = (1 << 63) + (instance << 40) + turn
            parameters = dict(epoch=epoch)
            ticket = call.submit(values=parameters, sample_gpu=turn == 0)
            assert ticket is not None
            parameters["epoch"] = 0
            tickets.append(ticket)
            expected.append((host, epoch))
        for ticket, arrays, (host, epoch) in zip(tickets, frames, expected):
            ticket.wait()
            assert read(arrays["out"], "Q") == [epoch, *[int(sum(x * x for x in lane)) for lane in host]]
            for lane in range(2):
                assert read(arrays[f"y{lane}"]) == [x * x for x in host[lane]]
                assert read(arrays[f"dx{lane}"]) == [4 * x for x in host[lane]]
            timing = ticket.gpu_timing
            assert timing["status"] == ("ready" if turn == 0 else "not_sampled")
            if turn == 0:
                assert timing["duration_ns"] > 0 and timing["scope"] == "execution_envelope"
                kept.append((ticket, timing))
            else:
                ticket.close()
    calls[0].close()
    with_ticket = calls[1].submit(values=dict(epoch=(1 << 64) - 1))
    assert with_ticket is not None
    # Retirement covers all tails, including the other domain's queued work.
    if borrowed:
        import taichi_forge as ti
        ti.reset()
    else:
        domains[1].close()
    assert with_ticket.done()
    with_ticket.close()
    try:
        calls[1].submit(values=dict(epoch=1))
    except (AotError, RuntimeError):
        pass
    else:
        raise AssertionError("Retired instance submitted work")
    calls[1].close()
    for domain in domains:
        domain.close()
    for ticket, timing in kept:
        assert ticket.done() and ticket.gpu_timing == timing
        ticket.close()
    print("VULKAN_PARALLEL_OK")


mode, artifact = sys.argv[1:]
if mode == "standalone":
    sys.meta_path.insert(0, NoFrontend())
    from taichi_forge_runtime.aot import Runtime
    runtime = Runtime("vulkan", device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", "0")))
else:
    import taichi_forge as ti
    ti.init(arch=ti.vulkan, offline_cache=False)
    runtime = ti.aot.borrow_runtime()
with runtime:
    check(runtime, artifact, mode == "borrowed")
