"""Catch repository-qualified include spelling errors even on Windows."""

from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _include_case_errors(tracked, read_source):
    # Git preserves directory spelling, including uninitialized submodule roots.
    # Missing generated headers and vendor-internal paths are left to the build.
    known = {}
    for name in filter(None, tracked):
        parts = PurePosixPath(name).parts
        for end in range(1, len(parts) + 1):
            prefix = "/".join(parts[:end])
            known.setdefault(prefix.casefold(), set()).add(prefix)

    include = re.compile(
        r'^[ \t]*#[ \t]*include[ \t]*(["<])([^">\n]+)[">]',
        re.M,
    )
    errors = []
    for name in filter(None, tracked):
        path = PurePosixPath(name)
        if path.parts[0] not in {"taichi", "c_api", "tests"}:
            continue
        if path.suffix not in {".h", ".hpp", ".cpp", ".cc", ".c", ".cu", ".cuh", ".mm"}:
            continue
        text = read_source(name)
        for match in include.finditer(text):
            target = match.group(2)
            candidates = [target]
            # Quoted includes can be relative to their source file. Angle
            # includes can also use repository-qualified include paths.
            if match.group(1) == '"':
                candidates.append((path.parent / target).as_posix())
            for candidate in candidates:
                parts = PurePosixPath(posixpath.normpath(candidate)).parts
                for end in range(1, len(parts) + 1):
                    prefix = "/".join(parts[:end])
                    alternatives = known.get(prefix.casefold())
                    if alternatives and prefix not in alternatives:
                        line = text.count("\n", 0, match.start()) + 1
                        errors.append(f"{name}:{line}: {prefix!r} must match {sorted(alternatives)!r}")
                        break
    return errors


def test_repository_includes_match_git_path_case():
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode("utf-8").split("\0")
    errors = _include_case_errors(
        tracked, lambda name: (ROOT / name).read_text(encoding="utf-8")
    )
    assert not errors, "Case-sensitive include mismatches:\n" + "\n".join(errors)


@pytest.mark.parametrize(
    "include, valid",
    [
        ('"taichi/detail/Api.h"', True),
        ('"taichi/detail/api.h"', False),
        ('<taichi/detail/Api.h>', True),
        ('<taichi/detail/api.h>', False),
        ('"../detail/Api.h"', True),
        ('"../detail/api.h"', False),
        ('"external/PicoSHA2/picosha2.h"', True),
        ('"external/picosha2/picosha2.h"', False),
        ('<vector>', True),
    ],
)
def test_include_case_audit_handles_relative_and_vendor_paths(include, valid):
    sources = {
        "taichi/example/main.cpp": f"#include {include}\n",
        "taichi/detail/Api.h": "",
        # Uninitialized submodules still carry their directory spelling in Git.
        "external/PicoSHA2": "",
    }
    errors = _include_case_errors(sources, sources.__getitem__)
    assert bool(errors) is not valid
