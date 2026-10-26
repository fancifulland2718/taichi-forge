"""Public current-version numbers must follow the native source contracts."""

from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[2]


def number(path, pattern):
    match = re.search(pattern, (ROOT / path).read_text(encoding="utf-8"))
    assert match, (path, pattern)
    return int(match.group(1))


@pytest.mark.parametrize("language", ["en", "zh"])
def test_release_notes_match_current_source_contracts(language):
    abi = number(
        "taichi/common/runtime_contract.h", r"kForgeNativeAbiRevision\s*=\s*(\d+)"
    )
    assert abi == number(
        "python/taichi_forge/_contract_constants.py",
        r"FORGE_NATIVE_ABI_REVISION\s*=\s*(\d+)",
    )
    plan = number("taichi/aot/execution_plan.h", r"kMaxVersion\s*=\s*(\d+)")
    cache = number(
        "taichi/analysis/offline_cache_util.h",
        r"kOfflineCacheSchemaVersion\s*=\s*(\d+)",
    )
    gfx = number("taichi/runtime/gfx/aot_utils.h", r"kMetadataVersion\s*=\s*(\d+)")
    text = (ROOT / f"docs/forge/release_notes.{language}.md").read_text(
        encoding="utf-8"
    )
    version = (ROOT / "version.txt").read_text().strip().removeprefix("v")
    section = re.search(
        rf"^## {re.escape(version)}[^\n]*\n(.*?)(?=^## |\Z)", text, re.M | re.S
    )
    assert section
    section = section.group(1)
    for expected, pattern in (
        (abi, r"private ABI is \*\*(\d+)\*\*|私有 ABI \*\*(\d+)\*\*"),
        (plan, r"execution-plan version \*\*(\d+)\*\*|持久执行计划为 \*\*(\d+)\*\*"),
        (
            cache,
            r"compiled-cache schema \*\*(\d+)\*\*|编译缓存 schema 为 \*\*(\d+)\*\*",
        ),
        (gfx, r"GFX metadata (?:为 )?\*\*(\d+)\*\*"),
    ):
        match = re.search(pattern, section)
        assert match, pattern
        assert int(next(value for value in match.groups() if value)) == expected


@pytest.mark.parametrize("language", ["en", "zh"])
def test_aot_pairing_guidance_uses_shared_current_version_reference(language):
    text = (ROOT / f"docs/forge/aot_sdk.{language}.md").read_text(encoding="utf-8")
    pairing = text.split("## ", 2)[1]
    assert f"release_notes.{language}.md#" in pairing
    assert "execution_plan.h" in pairing
    assert not re.search(
        r"current revision is \*\*\d+|当前修订为 \*\*\d+|currently up to \d+|当前最高为 \d+",
        pairing,
    )
