"""Bundled OptiX adapters in driver negotiation order. Discovery loads no libraries."""

import importlib.util
import os
from pathlib import Path


SUPPORTED_OPTIX_ABIS = (105, 118)


def bundled_optix_adapters():
    directories = []
    for name in ("taichi_forge_runtime", "taichi_forge"):
        spec = importlib.util.find_spec(name)
        if spec is not None:
            directories.extend(Path(root) / "_lib" / "hardware_providers"
                               for root in spec.submodule_search_locations or ())
    for abi in reversed(SUPPORTED_OPTIX_ABIS):
        stem = f"taichi_forge_optix_provider_abi1_optix{abi}"
        filename = f"{stem}.dll" if os.name == "nt" else f"lib{stem}.so"
        for directory in dict.fromkeys(directories):
            path = directory / filename
            if path.is_file():
                yield path
