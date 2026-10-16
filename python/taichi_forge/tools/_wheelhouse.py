"""Seal already resolved wheels into a relocatable standard pylock bundle.

Resolution and installation belong to standard packaging tools. This producer
only verifies the locked artifacts and replaces their locations with local paths.
It neither imports the DSL nor installs packages or downloads files.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import time
from urllib.parse import unquote, urlsplit


def seal(lock, wheels, output):
    import tomli
    import tomli_w
    from packaging.pylock import Pylock
    from packaging.utils import canonicalize_name, parse_wheel_filename
    from packaging.version import Version

    start = time.perf_counter()
    lock, wheels, output = Path(lock).resolve(), Path(wheels).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Wheelhouse output must be new; published artifacts are immutable")
    document = tomli.loads(lock.read_text(encoding="utf-8"))
    Pylock.from_dict(document)  # The public packaging implementation owns validation.
    if document["lock-version"] != "1.0" or not document["packages"]:
        raise ValueError("Expected a nonempty pylock 1.0 document")
    records = {}
    for package in document["packages"]:
        if any(key in package for key in ("sdist", "vcs", "directory", "archive")) or not package.get("wheels"):
            raise ValueError("A sealed wheelhouse requires wheel-only locked packages")
        for artifact in package["wheels"]:
            name = (
                artifact.get("name")
                or PurePosixPath(unquote(urlsplit(artifact.get("url", artifact.get("path", ""))).path)).name
            )
            if not name or PurePosixPath(name).name != name or "\\" in name or ":" in name:
                raise ValueError("Locked wheel name must be a plain filename")
            distribution, version, _, _ = parse_wheel_filename(name)
            if distribution != canonicalize_name(package["name"]) or version != Version(package["version"]):
                raise ValueError("Locked wheel name does not match its package and version")
            digest = artifact["hashes"].get("sha256", "")
            if len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
                raise ValueError("Locked wheels require a complete SHA256 digest")
            source = wheels / name
            if source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(wheels):
                raise ValueError("Locked wheel is missing or outside the wheel directory: " + name)
            identity = (digest.lower(), artifact.get("size"))
            if name in records and records[name][1:] != identity:
                raise ValueError("Conflicting locked wheel identity: " + name)
            records[name] = (source, *identity)
            artifact.pop("url", None)
            artifact["path"] = "wheels/" + name
    # Copy and hash the exact published bytes in one pass. A failed copy/hash
    # leaves no published directory; unrelated files in the cache are excluded.
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with tempfile.TemporaryDirectory(prefix=output.name + ".pending-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        (stage / "wheels").mkdir(parents=True)
        for name, (source, digest, size) in records.items():
            observed, count = hashlib.sha256(), 0
            with source.open("rb") as reader, (stage / "wheels" / name).open("xb") as writer:
                for block in iter(lambda: reader.read(1 << 20), b""):
                    writer.write(block)
                    observed.update(block)
                    count += len(block)
            if observed.hexdigest() != digest or (size is not None and count != size):
                raise ValueError("Locked wheel content differs from its hash or size: " + name)
            total += count
        (stage / "pylock.toml").write_bytes(tomli_w.dumps(document).encode("utf-8"))
        stage.rename(output)
    return dict(
        packages=len(document["packages"]),
        wheels=len(records),
        wheel_bytes=total,
        seal_ms=(time.perf_counter() - start) * 1000,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--wheels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(seal(args.lock, args.wheels, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
