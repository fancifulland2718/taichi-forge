"""Package sealed host extensions with standard Python metadata, without a DSL.

The producer uses PyPA build and setuptools. It never compiles host modules,
resolves runtime dependencies, imports application code, or bundles Forge.
"""

import argparse
import base64
import csv
import hashlib
import importlib.machinery
import io
import json
import keyword
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from zipfile import ZipFile


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _relative(value):
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
        or PurePosixPath(value).is_absolute()
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError(f"Expected a relative POSIX path: {value!r}")
    return value


def _inside(root, relative):
    path = root / _relative(relative)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing, linked or out-of-root input: {relative}")
    return path


def _name(name):
    if not isinstance(name, str) or not all(p.isidentifier() and not keyword.iskeyword(p) for p in name.split(".")):
        raise ValueError(f"Invalid host module name: {name!r}")
    return name


def _unique_json(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate manifest field: {name}")
        result[name] = value
    return result


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json)


def host_payload(layout):
    """Verify the entire sealed layout, including absence of undeclared files."""
    manifest = _load(_inside(layout, "host-manifest.json"))
    if manifest.get("version") != 1 or manifest.get("kind") != "forge_compiled_host":
        raise ValueError("Unsupported compiled host manifest")
    if (
        manifest.get("platform") != sysconfig.get_platform()
        or manifest.get("cache_tag") != sys.implementation.cache_tag
    ):
        raise ValueError("Build the host wheel with its matching Python ABI and platform")
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    modules, payload, names = [], {}, set()
    for item in manifest["modules"]:
        name, relative = _name(item["name"]), _relative(item["file"])
        stem = name.replace(".", "/")
        if relative == stem + suffix:
            extension = name
        elif relative == stem + "/__init__" + suffix:
            extension = name + ".__init__"
        else:
            raise ValueError(f"Host artifact does not match its module and ABI: {name}")
        if os.path.normcase(name) in names or os.path.normcase(relative) in payload:
            raise ValueError(f"Conflicting host module or artifact: {name}")
        names.add(os.path.normcase(name))
        source = _inside(layout, relative)
        data = source.read_bytes()
        if len(data) != item["bytes"] or _hash(data) != item["sha256"]:
            raise ValueError(f"Published host artifact changed: {name}")
        payload[os.path.normcase(relative)] = dict(file=relative, source=source, sha256=item["sha256"])
        modules.append(dict(name=extension, file=relative))
    if not modules:
        raise ValueError("A host wheel requires compiled modules")
    for alias, contract in manifest.get("aliases", {}).items():
        _name(alias)
        if os.path.normcase(alias) in names or set(contract) != {"target", "registered_by"}:
            raise ValueError(f"Invalid host alias contract: {alias}")
        names.add(os.path.normcase(alias))
        canonical = {item["name"] for item in manifest["modules"]}
        if contract["target"] not in canonical or contract["registered_by"] not in canonical:
            raise ValueError(f"Host alias has no compiled target or registrar: {alias}")
    actual = set()
    for path in layout.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Unexpected host layout symlink: {path}")
        if path.is_file():
            actual.add(path.relative_to(layout).as_posix())
    expected = {item["file"] for item in payload.values()} | {"host-manifest.json"}
    if actual != expected:
        raise ValueError("Unexpected host layout contents; publish only declared compiled artifacts")
    return manifest, modules, payload


def release_inputs(filename):
    filename = Path(filename).resolve()
    spec = _load(filename)
    if set(spec) != {"version", "host_layout", "project", "files"} or spec["version"] != 1:
        raise ValueError("Unsupported host wheel specification")
    root = filename.parent
    layout = (root / _relative(spec["host_layout"])).resolve()
    if not layout.is_relative_to(root):
        raise ValueError("Host layout is outside the release inputs")
    manifest, modules, payload = host_payload(layout)
    roots = {item["name"].split(".")[0] for item in manifest["modules"]}
    forbidden = {
        ".py",
        ".pyc",
        ".pyo",
        ".pyx",
        ".pxd",
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hpp",
        ".dll",
        ".dylib",
        ".lib",
        ".a",
        ".obj",
        ".o",
        ".pdb",
    }
    for target, relative in spec["files"].items():
        _relative(target)
        path = PurePosixPath(target)
        if path.parts[0] not in roots or len(path.parts) < 2:
            raise ValueError(f"Public files must belong to a compiled host namespace: {target}")
        if (
            path.suffix.lower() in forbidden
            or re.search(r"\.so(?:\.[0-9]+)*$", target.lower())
            or any(target.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)
        ):
            raise ValueError(f"Public files cannot introduce source or unverified extensions: {target}")
        key = os.path.normcase(target)
        if key in payload:
            raise ValueError(f"Conflicting wheel destination: {target}")
        source = _inside(root, relative)
        payload[key] = dict(file=target, source=source, sha256=_hash(source.read_bytes()))
    project = _inside(root, spec["project"])
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    config = tomllib.loads(project.read_text(encoding="utf-8"))
    build = config.get("build-system", {})
    if build.get("build-backend") != "setuptools.build_meta" or "backend-path" in build:
        raise ValueError("Closed host wheels use the standard setuptools.build_meta backend")
    if "setuptools" in config.get("tool", {}) or config.get("project", {}).get("dynamic"):
        raise ValueError("Closed host metadata must be static and omit source-discovery configuration")
    # Setuptools owns PEP 621 validation, dependency/extras normalization and
    # entry point generation. Only explicit metadata files are copied here.
    metadata = config.get("project", {})
    attachments = {}
    readme = metadata.get("readme")
    if isinstance(readme, str):
        attachments[readme] = _inside(project.parent, readme)
    elif isinstance(readme, dict) and "file" in readme:
        attachments[readme["file"]] = _inside(project.parent, readme["file"])
    license_value = metadata.get("license")
    if isinstance(license_value, dict) and "file" in license_value:
        attachments[license_value["file"]] = _inside(project.parent, license_value["file"])
    for pattern in metadata.get("license-files", []):
        _relative(pattern)
        matches = [p for p in project.parent.glob(pattern) if p.is_file()]
        if not matches:
            raise ValueError(f"License pattern has no files: {pattern}")
        for path in matches:
            name = path.relative_to(project.parent).as_posix()
            attachments[name] = _inside(project.parent, name)
    reserved = {"setup.py", "pyproject.toml", "host-input.json", "payload", "build", "dist"}
    for name in attachments:
        if PurePosixPath(name).parts[0] in reserved or PurePosixPath(name).suffix.lower() in forbidden:
            raise ValueError(f"Metadata file conflicts with the closed build inputs: {name}")
    return layout, manifest, modules, payload, project, attachments


# This build adapter copies already verified extension files. Setuptools owns
# wheel tags, metadata, RECORD and entry points; no application module is loaded.
_SETUP = """from pathlib import Path
import json
from setuptools import Distribution, Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.egg_info import egg_info

inputs = json.loads(Path("host-input.json").read_text(encoding="utf-8"))

class HostDistribution(Distribution):
    def has_pure_modules(self):
        return bool(inputs["public_files"])

class HostExtensions(build_ext):
    def run(self):
        for item in inputs["extensions"]:
            target = Path(self.get_ext_fullpath(item["name"]))
            target.parent.mkdir(parents=True, exist_ok=True)
            self.copy_file(str(Path("payload") / item["file"]), str(target))

class HostPublicFiles(build_py):
    def run(self):
        for name in inputs["public_files"]:
            target = Path(self.build_lib) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            self.copy_file(str(Path("payload") / name), str(target))
    def get_source_files(self):
        return [str(Path("payload") / name) for name in inputs["public_files"]]

class HostMetadata(egg_info):
    def run(self):
        super().run()
        (Path(self.egg_info) / "forge-host.json").write_text(
            json.dumps(inputs["manifest"], sort_keys=True, separators=(",", ":")), encoding="utf-8")

setup(distclass=HostDistribution, packages=[], py_modules=[],
      ext_modules=[Extension(item["name"], sources=[]) for item in inputs["extensions"]],
      cmdclass={"build_ext": HostExtensions, "build_py": HostPublicFiles, "egg_info": HostMetadata})
"""


def verify_wheel(path, payload):
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename

    tags = parse_wheel_filename(path.name)[3]
    if not tags or not tags <= set(sys_tags()) or any(tag.abi == "none" for tag in tags):
        raise ValueError("Host wheel has incompatible or missing native ABI tags")
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Wheel contains duplicate files")
        for name in names:
            _relative(name)
        metadata = {n.split("/", 1)[0] for n in names if ".dist-info/" in n}
        if len(metadata) != 1:
            raise ValueError("Wheel requires one metadata owner")
        metadata = metadata.pop()
        expected = {item["file"] for item in payload.values()}
        actual = {name for name in names if not name.startswith(metadata + "/")}
        if actual != expected:
            raise ValueError(f"Wheel payload differs from its release closure: {actual ^ expected}")
        if "Root-Is-Purelib: false" not in archive.read(metadata + "/WHEEL").decode():
            raise ValueError("Compiled host wheel was tagged as pure Python")
        archive.getinfo(metadata + "/forge-host.json")
        records = list(csv.reader(io.StringIO(archive.read(metadata + "/RECORD").decode())))
        if len(records) != len(names) or {row[0] for row in records} != set(names):
            raise ValueError("Wheel RECORD does not own every published file")
        for name, digest, size in records:
            if name.endswith(".dist-info/RECORD"):
                continue
            data = archive.read(name)
            actual_hash = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            if digest != "sha256=" + actual_hash or size != str(len(data)):
                raise ValueError(f"Wheel RECORD hash mismatch: {name}")
        for item in payload.values():
            if _hash(archive.read(item["file"])) != item["sha256"]:
                raise ValueError(f"Wheel changed a verified input: {item['file']}")


def build(spec, output):
    start = time.perf_counter()
    layout, manifest, modules, payload, project, attachments = release_inputs(spec)
    verified = time.perf_counter()
    output = Path(output).resolve()
    if output.is_relative_to(layout) or layout.is_relative_to(output):
        raise ValueError("Wheel output and sealed host layout must be separate")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="host-wheel-", dir=output.parent) as temporary:
        stage = Path(temporary)
        for item in payload.values():
            target = stage / "payload" / item["file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item["source"], target)
        for name, source in attachments.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        shutil.copyfile(project, stage / "pyproject.toml")
        (stage / "setup.py").write_text(_SETUP, encoding="utf-8")
        extension_files = {item["file"] for item in modules}
        (stage / "host-input.json").write_text(
            json.dumps(
                dict(
                    extensions=modules,
                    manifest=manifest,
                    public_files=sorted(
                        item["file"] for item in payload.values() if item["file"] not in extension_files
                    ),
                )
            ),
            encoding="utf-8",
        )
        environment = os.environ.copy()
        # A checkout may contain build.py (Forge does). Neither that module nor
        # producer PYTHONPATH entries belong to the standard packaging process.
        environment.pop("PYTHONPATH", None)
        environment["PIP_NO_INDEX"] = "1"
        environment.setdefault("SOURCE_DATE_EPOCH", "315532800")
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(stage / "dist"),
                str(stage),
            ],
            env=environment,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode:
            raise RuntimeError("Host wheel build failed:\n" + result.stdout + result.stderr)
        wheels = list((stage / "dist").glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("Host packaging must produce exactly one wheel")
        packaged = time.perf_counter()
        verify_wheel(wheels[0], payload)
        destination = output / wheels[0].name
        if destination.exists():
            if destination.read_bytes() != wheels[0].read_bytes():
                raise ValueError("Published wheel is immutable; use a new output directory or version")
        else:
            wheels[0].rename(destination)
    return dict(
        wheel=str(destination),
        bytes=destination.stat().st_size,
        modules=len(modules),
        payload_files=len(payload),
        input_verify_ms=(verified - start) * 1000,
        package_ms=(packaged - verified) * 1000,
        output_verify_ms=(time.perf_counter() - packaged) * 1000,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.spec, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
