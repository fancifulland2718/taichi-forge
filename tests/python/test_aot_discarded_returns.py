"""CGraphs and fixed plans discard results without dropping work or dependencies."""

import os
from pathlib import Path
import subprocess
import sys

import taichi_forge as ti
from taichi_forge._lib import utils
from tests import test_utils
from tests.cpp.aot.python_scripts.discarded_return_aot import export


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_cgraph_discarded_returns_gradients_and_retirement(tmp_path):
    arch = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    export(tmp_path)
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=os.pathsep.join([str(root / "packaging/runtime/src"), *sys.path]),
        TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
        TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir(),
        TI_ENABLE_VERSION_CHECK="0",
    )
    for mode, execution in ((mode, execution) for mode in ("standalone", "borrowed")
                            for execution in ("ordinary", "prepared", "plan")):
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                str(Path(__file__).with_name("aot_discarded_return_probe.py")),
                mode,
                arch,
                str(tmp_path),
                execution,
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=90,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert not any(
            item in output for item in ("VUID-", "Validation Error", "SYNC-HAZARD")
        ), output


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_discarded_return_scope_isolation_and_initialization(tmp_path):
    import array
    from taichi_forge_runtime.aot import ExecutionPlan, Runtime

    @ti.kernel
    def advance(state: ti.types.ndarray(ti.i32, 1)) -> ti.i32:
        state[0] += 1
        return state[0]

    module = ti.aot.Module()
    module.add_kernel(advance)
    module.save(tmp_path)
    arch = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    with Runtime(arch, device_index=int(os.environ.get("TI_TEST_VULKAN_DEVICE_INDEX", 0))
                 if arch == "vulkan" else 0) as runtime:
        loaded = runtime.load_module(tmp_path)
        arrays = {name: runtime.ndarray("i32", 1) for name in ("a", "b")}
        def read(name):
            return array.array("i", arrays[name].read())[0]
        def step(name):
            return dict(kind="kernel", entry="advance", arguments=[name])
        serial = ExecutionPlan([step("a"), step("a"), step("b")],
                               initialization=[step("a")])
        for value in arrays.values():
            value.write(bytes(4))
        first = loaded.prepare_execution(serial, arrays)
        assert first.info["native_resource_bytes"] == 8
        assert read("a") == 0  # Preparation must not execute the initializer.
        first.initialize()
        assert first.info["native_resource_bytes"] == 4
        assert read("a") == 1
        first.run()
        first.close()  # In-flight scratch retires after its recording.
        assert (read("a"), read("b")) == (3, 1)
        if arch == "cuda":
            parallel = ExecutionPlan([
                dict(kind="cuda_parallel", branches=[[step("a"), step("a")],
                                                      [step("b"), step("b")]]),
                step("a"),
            ])
            call = loaded.prepare_execution(parallel, arrays)
            # Two concurrent branches plus the sequential suffix have distinct
            # frames, even though every kernel has the same return layout.
            assert call.info["native_resource_bytes"] == 12
            for _ in range(3):
                call.run()
            call.close()
            assert (read("a"), read("b")) == (12, 7)
            domain = runtime.execution_domain()
            selected = loaded.prepare_execution(serial, arrays, execution_domain=domain)
            selected.initialize()
            ticket = selected.submit()
            selected.close()
            ticket.wait()
            assert (read("a"), read("b")) == (15, 8)
            ticket.close()
            domain.close()
