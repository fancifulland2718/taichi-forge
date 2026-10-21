"""Persistent packs share the native registry and existing array/adjoint ABI."""

import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import pytest

import taichi_forge as ti
from taichi_forge._lib import utils
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_persistent_argpack_values_private_arrays_and_adjoint(tmp_path):
    child = ti.types.argpack(bias=ti.f32)
    pack_type = ti.types.argpack(gain=ti.f32, child=child,
                                 values=ti.types.ndarray(ti.f32, 1, needs_grad=True))
    resource_only = ti.types.argpack(values=ti.types.ndarray(ti.f32, 1))

    @ti.kernel
    def energy(p: pack_type, loss: ti.types.ndarray(ti.f32, 1, needs_grad=True)):
        for i in p.values:
            loss[0] += p.gain * p.values[i] ** 2 + p.child.bias * p.values[i]

    @ti.kernel
    def advance(values: ti.types.ndarray(ti.f32, 1)):
        for i in values:
            values[i] += 1

    @ti.kernel
    def copy_pack(p: resource_only, output: ti.types.ndarray(ti.f32, 1)):
        for i in p.values:
            output[i] = p.values[i]

    start = time.perf_counter()
    module = ti.aot.Module()
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="backward")
    module.add_kernel(advance)
    module.add_kernel(copy_pack)
    layout = module.get_argpack_layout("energy", 0)
    source = next(member["origin"] for member in layout["members"] if member["origin"]["kind"] == "ndarray")
    assert source["logical"] == [0, 2] and source["physical"] == [2]
    with pytest.raises(ValueError, match="exactly match"):
        module.argpack_resources("invalid", "energy", 0, dict(gain=2, child=dict(bias=1), values="state"))
    initial = module.add_data(struct.pack("3f", 1, -2, 3))
    seed = module.add_data(struct.pack("f", 1))

    def make_plan(gain, bias):
        resources = [
            dict(kind="ndarray", name="state", dtype="f32", shape=[3], initial_data=initial),
            dict(kind="ndarray", name="state_grad", dtype="f32", shape=[3]),
            dict(kind="ndarray", name="seed", dtype="f32", shape=[1], initial_data=seed),
        ]
        resources.extend(module.argpack_resources("pack", "energy", 0,
                                                  dict(gain=gain, child=dict(bias=bias))))
        return ti.aot.ExecutionPlan([
            dict(kind="fill32", destination="loss", value=0),
            dict(kind="fill32", destination="state_grad", value=0),
            # Resource members already occupy physical kernel parameters.
            dict(kind="kernel", entry="energy", arguments=["pack", "loss", "state"]),
            dict(kind="kernel", entry="backward", arguments=["pack", "loss", "state"]),
            dict(kind="copy", source="state_grad", destination="gradient"),
            dict(kind="kernel", entry="advance", arguments=["state"]),
        ], resources=resources, gradients=dict(state="state_grad", loss="seed"))

    module.add_execution_plan("step", make_plan(2, 1))
    module.add_execution_plan("other_values", make_plan(3, -1))
    module.add_execution_plan("resource_only", ti.aot.ExecutionPlan([
        dict(kind="kernel", entry="copy_pack", arguments=["pack", "output", "state"]),
    ], resources=[
        dict(kind="ndarray", name="state", dtype="f32", shape=[3], initial_data=initial),
        *module.argpack_resources("pack", "copy_pack", 0, {}),
    ]))
    artifact = tmp_path / "persistent-packs.tcm"
    module.archive(str(artifact))
    print(json.dumps(dict(export_seconds=time.perf_counter() - start, artifact_bytes=artifact.stat().st_size)))
    arch = "cpu" if ti.cfg.arch == ti.cpu else ti.cfg.arch.name
    environment = dict(os.environ,
                       PYTHONPATH=os.pathsep.join(sys.path),
                       TAICHI_NATIVE_RUNTIME_DIR=str(Path(utils._loaded_native_runtime_path).parent),
                       TAICHI_RUNTIME_DIR=utils._runtime_bitcode_dir())
    ti.reset()  # Consumers cannot borrow producer packs, compiled handles or storage.
    for mode in ("standalone", "borrowed"):
        result = subprocess.run(
            [sys.executable, *(["-S"] if mode == "standalone" else []),
             str(Path(__file__).with_name("aot_argpack_persistent_probe.py")), mode, arch, str(artifact)],
            env=environment, capture_output=True, text=True, timeout=90,
        )
        output = result.stdout + result.stderr
        print(output)
        assert result.returncode == 0, output
        assert "PERSISTENT_ARGPACK_PASSED" in output
        assert not any(token in output for token in ("VUID-", "Validation Error", "SYNC-HAZARD"))
