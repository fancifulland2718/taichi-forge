"""Export the resident Vulkan JIT specialization without submitting it."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl


def vulkan_runtime_diagnostics():
    """Return device/driver identity and the current runtime's cache provenance.

    ``supplied_to_backend`` means cache bytes were passed to Vulkan. Drivers may
    silently ignore incompatible data; this is not a cache-hit counter.
    """
    runtime = impl.get_runtime()
    if runtime.prog is None or impl.current_cfg().arch != _ti_core.vulkan:
        raise RuntimeError("Vulkan diagnostics require ti.init(arch=ti.vulkan)")
    runtime.materialize()
    raw = runtime.prog._vulkan_runtime_diagnostics()
    return {
        "device": json.loads(raw["device_json"]),
        "pipeline_cache": dict(raw["pipeline_cache"]),
        "forge_version": _ti_core.get_version_string(),
        "forge_commit": _ti_core.get_commit_hash(),
    }


def export_vulkan_kernel(kernel, *args, directory, validate=False):
    """Write the actual JIT SPIR-V, layout and resolved configuration to a new directory.

    Arguments specialize the kernel but are not executed or copied to the host.
    ``validate=True`` requires the SDK's ``spirv-val`` on PATH. Pipeline-time
    validation instead uses the bundled validator with TI_VULKAN_VALIDATE_SPIRV=1.
    AOT output and GPU execution are separate operations.
    """
    from taichi_forge.lang.kernel_impl import Kernel, _BoundedDifferentiableMethod

    runtime_info = vulkan_runtime_diagnostics()
    primal = getattr(kernel, "_primal", kernel)
    if not isinstance(primal, Kernel):
        raise TypeError("export_vulkan_kernel expects a @ti.kernel function")
    owner = getattr(kernel, "__self__", None)
    if isinstance(kernel, _BoundedDifferentiableMethod) and not kernel._is_staticmethod:
        owner = kernel._kernel_owner
    if owner is not None:
        args = (owner, *args)
    validator = shutil.which("spirv-val") if validate else None
    if validate and validator is None:
        raise RuntimeError("validate=True requires spirv-val on PATH")
    key = primal.ensure_compiled(*args)
    raw = impl.get_runtime().prog._kernel_vulkan_artifacts(primal.compiled_kernels[key])
    config = raw["resolved_config"]
    config_values = {"arch": "vulkan"}
    for name in dir(config):
        if not name.startswith("_"):
            value = getattr(config, name)
            if isinstance(value, (bool, int, float, str)):
                config_values[name] = value
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "kind": "vulkan_jit",
        "kernel_identity": raw["kernel_identity"],
        "pipeline_registered": raw["pipeline_registered"],
        "metadata": json.loads(raw["metadata_json"]),
        "resolved_config": config_values,
        "runtime": runtime_info,
        "specialization_constants": [],
        "shaders": [],
    }
    failed = False
    for ordinal, binary in enumerate(raw["shaders"]):
        path = destination / f"shader-{ordinal}.spv"
        path.write_bytes(binary)
        shader = {
            "file": path.name,
            "sha256": hashlib.sha256(binary).hexdigest(),
            "bytes": len(binary),
        }
        if validator:
            version = runtime_info["device"]["api_version"]
            minor = min(3, (version >> 12) & 1023)
            command = [validator, "--target-env", f"vulkan1.{minor}", str(path)]
            check = subprocess.run(command, capture_output=True, text=True, check=False)
            shader["validation"] = {
                "command": command,
                "returncode": check.returncode,
                "stdout": check.stdout,
                "stderr": check.stderr,
            }
            failed |= check.returncode != 0
        manifest["shaders"].append(shader)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if failed:
        raise RuntimeError(
            f"SPIR-V validation failed; artifacts preserved in {destination}"
        )
    return manifest


__all__ = ["export_vulkan_kernel", "vulkan_runtime_diagnostics"]
