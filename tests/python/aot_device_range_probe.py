"""Consume selected range policies with the normally installed runtime only."""
import sys
import os
import json
import tempfile
import zipfile
from pathlib import Path
import numpy as np
from taichi_forge_runtime.aot import Runtime


with Runtime("vulkan", device_index=int(os.environ.get("TI_VISIBLE_DEVICE", "0"))) as runtime:
    module = runtime.load_module(sys.argv[1])
    if sys.argv[1].endswith("gradient.tcm"):
        arrays = {name: runtime.ndarray("f32", (65,)) for name in ("x", "y", "dx", "dy")}
        arrays["limits"] = runtime.ndarray("i32", (1,))
        arrays["x"].write((np.arange(65, dtype=np.float32) + 1).tobytes())
        call = module.prepare_execution("gradient", arrays)
        for size in (0, 5, 65, 0):
            arrays["limits"].write(np.array([size], dtype=np.int32).tobytes())
            arrays["y"].write(np.full(65, -1, dtype=np.float32).tobytes())
            arrays["dx"].write(np.zeros(65, dtype=np.float32).tobytes())
            arrays["dy"].write(np.ones(65, dtype=np.float32).tobytes())
            call()
            expected = np.zeros(65, dtype=np.float32)
            expected[:size] = 2 * (np.arange(size) + 1)
            np.testing.assert_array_equal(np.frombuffer(arrays["dx"].read(), dtype=np.float32), expected)
            expected_y = np.full(65, -1, dtype=np.float32)
            expected_y[:size] = (np.arange(size) + 1) ** 2
            np.testing.assert_array_equal(np.frombuffer(arrays["y"].read(), dtype=np.float32), expected_y)
        call.close()
    else:
        owners, calls = [], []
        for entry in ("fixed", "device", "again"):
            arrays = dict(out=runtime.ndarray("i32", (66,)), limits=runtime.ndarray("i32", (1,)))
            owners.append(arrays)
            calls.append(module.prepare_execution(entry, arrays))
        assert calls[1].info["workspace_bytes"] >= calls[0].info["workspace_bytes"] + 24
        print("RANGE-WORKSPACE", {entry: call.info["workspace_bytes"]
                                 for entry, call in zip(("fixed", "device", "again"), calls)})
        for size in (0, 5, 65, 0):
            for arrays, call in zip(owners, calls):
                arrays["out"].write(np.full(66, -1, dtype=np.int32).tobytes())
                arrays["limits"].write(np.array([size], dtype=np.int32).tobytes())
                call()
                expected = np.full(66, -1, dtype=np.int32)
                expected[:size] = np.arange(1, size + 1)
                expected[:size // 2] += 100
                expected[65] = 17
                np.testing.assert_array_equal(np.frombuffer(arrays["out"].read(), dtype=np.int32), expected)
        calls[1].close()
        owners[2]["limits"].write(np.array([5], dtype=np.int32).tobytes())
        calls[2]()
        np.testing.assert_array_equal(np.frombuffer(owners[2]["out"].read(), dtype=np.int32)[:5],
                                      [101, 102, 3, 4, 5])
        for call in calls:
            call.close()
        # A corrupted selection must not turn into an ordinary direct launch.
        with zipfile.ZipFile(sys.argv[1]) as archive:
            files = {name: archive.read(name) for name in archive.namelist()}
        original = json.loads(files["metadata.json"])
        for defect in ("version", "missing_version", "task", "duplicate", "unknown", "legacy",
                       "version_wrap", "index_wrap", "float_index", "negative_index"):
            metadata = json.loads(json.dumps(original))
            entry = metadata["device_range_dispatch"][0]
            if defect == "version":
                entry["version"] = 99
            elif defect == "missing_version":
                del entry["version"]
            elif defect == "task":
                entry["tasks"] = [0]
            elif defect == "duplicate":
                entry["tasks"] *= 2
            elif defect == "unknown":
                entry["name"] = "unknown"
            elif defect == "version_wrap":
                entry["version"] += 1 << 32
            elif defect == "index_wrap":
                entry["tasks"][0] += 1 << 32
            elif defect == "float_index":
                entry["tasks"][0] += 0.5
            elif defect == "negative_index":
                entry["tasks"][0] = -1
            else:
                metadata["metadata_version"] = 10
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.tcm"
                with zipfile.ZipFile(path, "w") as archive:
                    for name, data in files.items():
                        archive.writestr(name, json.dumps(metadata).encode() if name == "metadata.json" else data)
                try:
                    invalid = runtime.load_module(str(path))
                except (ValueError, RuntimeError):
                    pass
                else:
                    invalid.close()
                    raise AssertionError("Accepted corrupted range selection: " + defect)
    module.close()
assert "taichi_forge" not in sys.modules
print("RANGE-AOT runtime-only selected policy and state checks passed")
