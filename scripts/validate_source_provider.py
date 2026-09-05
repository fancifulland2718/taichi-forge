"""Audit an explicitly built CUB addon without loading Forge or CUDA.

This is a build/CI tool, never an import-time or replay-time check. It reuses
the manifest parser and the runtime wheel's binary dependency audit.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

if __package__:
    from .validate_runtime_wheel import _binary_imports, _validate_binary_dependencies
else:
    from validate_runtime_wheel import _binary_imports, _validate_binary_dependencies


def audit(manifest_path, platform):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_forge_source_manifest_audit",
        root / "python/taichi_forge/hardware/_source_provider.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    manifest = module.load_source_provider_manifest(
        manifest_path,
        expected_provider_id="cub_reference",
        expected_provider_abi="taichi-forge-cub-source-provider-c-abi1",
    )
    _validate_binary_dependencies(manifest.binary_path, platform)
    return {
        "build": manifest.build_report(),
        "binary_imports": sorted(_binary_imports(manifest.binary_path, platform)),
        "platform": platform,
        "execution_qualified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--platform", choices=("windows", "linux"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.manifest, args.platform)
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Audited {args.manifest}: {report['build']['build_identity']}")


if __name__ == "__main__":
    main()
