"""Runtime-only CUDA consumer: entry policy, gradients and independent owners."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np
from taichi_forge_runtime.aot import Runtime


artifact, mode = sys.argv[1:]


def array(owner, name, dtype, count):
    owner[name] = runtime.ndarray(dtype, (count,))


def reset(owner, size, lane=0):
    owner["limits"].write(np.array([size], dtype=np.int32).tobytes())
    if mode == "gradient":
        owner["x"].write((np.arange(65, dtype=np.float32) + lane + 1).tobytes())
        for name, value in (("y", -1), ("dx", 0), ("dy", 1)):
            owner[name].write(np.full(65, value, dtype=np.float32).tobytes())
    else:
        owner["out"].write(np.full(66, -1, dtype=np.int32).tobytes())


def check(owner, size, lane=0):
    if mode == "gradient":
        expected = np.full(65, -1, dtype=np.float32)
        expected[:size] = (np.arange(size) + lane + 1) ** 2
        np.testing.assert_array_equal(np.frombuffer(owner["y"].read(), dtype=np.float32), expected)
        expected = np.zeros(65, dtype=np.float32)
        expected[:size] = 2 * (np.arange(size) + lane + 1)
        np.testing.assert_array_equal(np.frombuffer(owner["dx"].read(), dtype=np.float32), expected)
    else:
        expected = np.full(66, -1, dtype=np.int32)
        expected[:size] = np.arange(size) + 1
        expected[:size // 2] += 100
        expected[65] = 17
        np.testing.assert_array_equal(np.frombuffer(owner["out"].read(), dtype=np.int32), expected)


def make_arrays():
    owner = {}
    array(owner, "limits", "i32", 1)
    for name in (("x", "y", "dx", "dy") if mode == "gradient" else ("out",)):
        array(owner, name, "f32" if mode == "gradient" else "i32", 65 if mode == "gradient" else 66)
    return owner


def entry_hash(metadata):
    compact = lambda item: json.dumps(item, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256((metadata["abi_sha256"] + "\n" + compact(metadata["kernel_aliases"])
                           + "\n" + compact(metadata["device_range_dispatch"])).encode()).hexdigest()


with Runtime("cuda") as runtime:
    module = runtime.load_module(artifact)
    entries = ("gradient",) if mode == "gradient" else ("fixed", "device", "again")
    workspaces = {}
    for entry in entries:
        owner = make_arrays()
        call = module.prepare_execution(entry, owner)
        workspaces[entry] = call.info["workspace_bytes"]
        for size in (0, 5, 65, 0):
            reset(owner, size)
            call()
            check(owner, size)
        call.close()
    if mode != "gradient":
        assert workspaces["device"] == workspaces["fixed"] + 96, workspaces
    print("ORDINARY-WORKSPACE", workspaces)

    domains = [runtime.execution_domain(instance_capacity=1) for _ in range(2)]
    owners = [make_arrays() for _ in domains]
    entry = "gradient" if mode == "gradient" else "device"
    calls = [module.prepare_execution(entry, owner, execution_domain=domain)
             for domain, owner in zip(domains, owners)]
    for call in calls:
        call.initialize()
    print("INDEPENDENT-WORKSPACE", [call.info["workspace_bytes"] for call in calls])
    for sizes in ((0, 65), (65, 5), (5, 0)):
        for lane, size in enumerate(sizes):
            reset(owners[lane], size, lane)
        runtime.wait()  # Explicit dependency on ordinary-queue test input writes.
        completions = [call.submit() for call in calls]
        for completion in completions:
            completion.wait()
            completion.close()
        for lane, size in enumerate(sizes):
            check(owners[lane], size, lane)
    calls[0].close()
    domains[0].close()
    reset(owners[1], 65, 1)
    runtime.wait()
    completion = calls[1].submit()
    completion.wait()
    completion.close()
    check(owners[1], 65, 1)
    calls[1].close()
    domains[1].close()

    if mode != "gradient":
        with zipfile.ZipFile(artifact) as archive:
            files = {name: archive.read(name) for name in archive.namelist()}
        original = json.loads(files["aot_metadata.json"])
        assert entry_hash(original) == original["kernel_entry_sha256"]
        # Recompute integrity after semantic mutations. Otherwise a checksum
        # failure alone would leave the actual reference/proof validators untested.
        defects = ("alias_target", "alias_chain", "alias_collision", "version", "missing_version",
                   "duplicate_entry", "duplicate_task", "unknown_entry", "unproven", "out_of_range",
                   "version_wrap", "index_wrap", "float_index", "negative_index", "legacy")
        for defect in defects:
            metadata = copy.deepcopy(original)
            alias = metadata["kernel_aliases"][0]
            selected = metadata["device_range_dispatch"][0]
            if defect == "alias_target":
                alias["value"] = "missing"
            elif defect == "alias_chain":
                alias["value"] = alias["key"]
            elif defect == "alias_collision":
                alias["key"] = alias["value"]
            elif defect == "version":
                selected["version"] = 99
            elif defect == "missing_version":
                del selected["version"]
            elif defect == "duplicate_entry":
                metadata["device_range_dispatch"].append(copy.deepcopy(selected))
            elif defect == "duplicate_task":
                selected["tasks"] *= 2
            elif defect == "unknown_entry":
                selected["name"] = "missing"
            elif defect == "unproven":
                selected["tasks"] = [0]
            elif defect == "out_of_range":
                selected["tasks"] = [0xFFFFFFFF]
            elif defect == "version_wrap":
                selected["version"] += 1 << 32
            elif defect == "index_wrap":
                selected["tasks"][0] += 1 << 32
            elif defect == "float_index":
                selected["tasks"][0] += 0.5
            elif defect == "negative_index":
                selected["tasks"][0] = -1
            else:
                metadata["schema_version"] = 11
            metadata["kernel_entry_sha256"] = entry_hash(metadata)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.tcm"
                with zipfile.ZipFile(path, "w") as archive:
                    for name, data in files.items():
                        archive.writestr(name, json.dumps(metadata).encode() if name == "aot_metadata.json" else data)
                try:
                    invalid = runtime.load_module(str(path))
                except (ValueError, RuntimeError):
                    pass
                else:
                    invalid.close()
                    raise AssertionError("Accepted invalid LLVM entry policy: " + defect)
        print("SEMANTIC-REJECTIONS", len(defects))
    module.close()
assert "taichi_forge" not in sys.modules
print("CUDA-RANGE-AOT runtime-only checks passed")
