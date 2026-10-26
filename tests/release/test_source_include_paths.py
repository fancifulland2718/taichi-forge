"""Catch repository-qualified include spelling errors even on Windows."""

from pathlib import Path, PurePosixPath
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def test_repository_qualified_includes_match_git_path_case():
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode("utf-8").split("\0")
    # Git preserves directory spelling, including uninitialized submodule roots.
    # Missing generated headers and vendor-internal paths are left to the build.
    known = {}
    for name in filter(None, tracked):
        parts = PurePosixPath(name).parts
        for end in range(1, len(parts) + 1):
            prefix = "/".join(parts[:end])
            known.setdefault(prefix.casefold(), set()).add(prefix)

    include = re.compile(
        r'^[ \t]*#[ \t]*include[ \t]*"((?:taichi|external|c_api)/[^"\n]+)"',
        re.M | re.I,
    )
    errors = []
    for name in filter(None, tracked):
        path = PurePosixPath(name)
        if path.parts[0] not in {"taichi", "c_api", "tests"}:
            continue
        if path.suffix not in {".h", ".hpp", ".cpp", ".cc", ".c", ".cu", ".cuh", ".mm"}:
            continue
        text = (ROOT / name).read_text(encoding="utf-8")
        for match in include.finditer(text):
            parts = PurePosixPath(match.group(1)).parts
            for end in range(1, len(parts) + 1):
                prefix = "/".join(parts[:end])
                alternatives = known.get(prefix.casefold())
                if alternatives and prefix not in alternatives:
                    line = text.count("\n", 0, match.start()) + 1
                    errors.append(f"{name}:{line}: {prefix!r} must match {sorted(alternatives)!r}")
                    break
    assert not errors, "Case-sensitive include mismatches:\n" + "\n".join(errors)
