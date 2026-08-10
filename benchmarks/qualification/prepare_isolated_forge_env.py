from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Sequence


NEUTRAL_DISTRIBUTIONS = (
    "numpy",
    "colorama",
    "dill",
    "rich",
    "markdown-it-py",
    "mdurl",
    "pygments",
)


def _normalized(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _site_packages(environment: Path) -> Path:
    return environment / "Lib" / "site-packages"


def _python(environment: Path) -> Path:
    return environment / "Scripts" / "python.exe"


def _distribution_map(site_packages: Path) -> dict[str, metadata.Distribution]:
    return {
        _normalized(distribution.metadata["Name"]): distribution
        for distribution in metadata.distributions(path=[str(site_packages)])
        if distribution.metadata["Name"]
    }


def _copy_distribution(distribution: metadata.Distribution,
                       source_site: Path, target_site: Path) -> int:
    copied = 0
    for entry in distribution.files or ():
        source = Path(distribution.locate_file(entry)).resolve()
        try:
            relative = source.relative_to(source_site.resolve())
        except ValueError:
            continue
        if not source.is_file():
            continue
        destination = target_site / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
    if copied == 0:
        raise RuntimeError(
            f"distribution {distribution.metadata['Name']} had no in-environment files")
    return copied


def _run(command: Sequence[str], environment: dict[str, str] | None = None) -> None:
    subprocess.run(list(command), check=True, env=environment)


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Build a dependency-complete Forge venv without network access")
    parser.add_argument(
        "--base-python",
        default=(r"C:\Users\Administrator\AppData\Local\Programs\Python"
                 r"\Python310\python.exe"))
    parser.add_argument(
        "--dependency-environment",
        default=str(repo_root / "temp_outputs" / "benchmark_envs" /
                    "vanilla-py310"))
    parser.add_argument(
        "--target-environment",
        default=str(repo_root / "temp_outputs" / "benchmark_envs" /
                    "forge-wheel-isolated-py310"))
    parser.add_argument(
        "--forge-shim-wheel",
        default=str(repo_root / "dist" /
                    "taichi_forge-0.6.2-cp310-cp310-win_amd64.whl"))
    parser.add_argument(
        "--forge-runtime-wheel",
        default=str(repo_root / "dist" /
                    "taichi_forge_runtime-0.6.2-py3-none-win_amd64.whl"))
    args = parser.parse_args(argv)

    base_python = Path(args.base_python).resolve()
    dependency_environment = Path(args.dependency_environment).resolve()
    target_environment = Path(args.target_environment).resolve()
    wheels = [Path(args.forge_shim_wheel).resolve(),
              Path(args.forge_runtime_wheel).resolve()]
    for path in (base_python, _python(dependency_environment), *wheels):
        if not path.is_file():
            raise FileNotFoundError(path)
    if target_environment.exists():
        raise FileExistsError(
            f"refusing to overwrite existing environment: {target_environment}")

    _run([str(base_python), "-m", "venv", str(target_environment)])
    target_python = _python(target_environment)
    _run([
        str(target_python), "-m", "pip", "install",
        "--disable-pip-version-check", "--no-index", "--no-deps",
        *(str(wheel) for wheel in wheels),
    ])

    source_site = _site_packages(dependency_environment)
    target_site = _site_packages(target_environment)
    available = _distribution_map(source_site)
    copied = {}
    for requested in NEUTRAL_DISTRIBUTIONS:
        key = _normalized(requested)
        if key not in available:
            raise RuntimeError(f"missing neutral dependency {requested} in {source_site}")
        distribution = available[key]
        copied[requested] = {
            "version": distribution.version,
            "file_count": _copy_distribution(distribution, source_site, target_site),
        }

    child_environment = os.environ.copy()
    child_environment.pop("PYTHONPATH", None)
    child_environment.pop("PYTHONHOME", None)
    child_environment["PYTHONNOUSERSITE"] = "1"
    validation_code = (
        "import importlib.metadata as m, json, pathlib, sys; "
        "names=['taichi-forge','taichi-forge-runtime','numpy','colorama','dill',"
        "'rich','markdown-it-py','mdurl','pygments']; "
        "import taichi_forge as ti; "
        "print(json.dumps({'python':sys.version,'prefix':sys.prefix,"
        "'package':str(pathlib.Path(ti.__file__).resolve()),"
        "'core':str(pathlib.Path(ti._lib.core.__file__).resolve()),"
        "'versions':{name:m.version(name) for name in names}},sort_keys=True))")
    completed = subprocess.run(
        [str(target_python), "-I", "-c", validation_code],
        check=True,
        env=child_environment,
        capture_output=True,
        text=True,
    )
    validation = json.loads(completed.stdout.splitlines()[-1])
    prefix = target_environment.resolve()
    for name in ("package", "core"):
        Path(validation[name]).resolve().relative_to(prefix)

    lock = {
        "schema": "taichi_forge.isolated_benchmark_environment.v1",
        "base_python": str(base_python),
        "dependency_source_environment": str(dependency_environment),
        "target_environment": str(target_environment),
        "neutral_distributions": copied,
        "wheels": [
            {"path": str(wheel), "sha256": _sha256(wheel)} for wheel in wheels
        ],
        "validation": validation,
        "network_used": False,
        "cross_environment_runtime_path_used": False,
    }
    (target_environment / "qualification-environment-lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(target_environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
