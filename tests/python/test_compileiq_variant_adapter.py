import sys
from types import ModuleType, SimpleNamespace

import pytest

from taichi_forge.lang._compileiq_adapter import _CompileIQVariantAdapter


class _FakeSession:
    def __init__(self):
        self._variants = {
            "v0-auto": SimpleNamespace(
                variant_id="v0-auto",
                compilation_id="c0",
                spec=SimpleNamespace(stable_payload="spec-v0-auto"),
            ),
            "v0-one": SimpleNamespace(
                variant_id="v0-one",
                compilation_id="c0",
                spec=SimpleNamespace(stable_payload="spec-v0-one"),
            ),
            "v1-auto": SimpleNamespace(
                variant_id="v1-auto",
                compilation_id="c1",
                spec=SimpleNamespace(stable_payload="spec-v1-auto"),
            ),
        }
        self.compilation_groups = (
            SimpleNamespace(
                compilation_id="c0",
                representative_variant_id="v0-auto",
                variant_ids=("v0-auto", "v0-one"),
            ),
            SimpleNamespace(
                compilation_id="c1",
                representative_variant_id="v1-auto",
                variant_ids=("v1-auto",),
            ),
        )

    def variant_ids(self):
        return tuple(self._variants)

    def variant(self, variant_id):
        return self._variants[variant_id]

    def bind(self, variant_id):
        return ("bound", variant_id)


def _install_fake_compileiq(monkeypatch):
    package = ModuleType("compileiq")
    search_spaces = ModuleType("compileiq.search_spaces")
    base = ModuleType("compileiq.search_spaces.base")
    compilers = ModuleType("compileiq.search_spaces.compilers")

    base.choice = lambda values: ("choice", tuple(values))

    class PtxasSearchSpace:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    compilers.PtxasSearchSpace = PtxasSearchSpace
    monkeypatch.setitem(sys.modules, "compileiq", package)
    monkeypatch.setitem(sys.modules, "compileiq.search_spaces", search_spaces)
    monkeypatch.setitem(sys.modules, "compileiq.search_spaces.base", base)
    monkeypatch.setitem(sys.modules, "compileiq.search_spaces.compilers", compilers)


def test_compileiq_adapter_keeps_structural_and_launch_stages_separate(monkeypatch):
    _install_fake_compileiq(monkeypatch)
    adapter = _CompileIQVariantAdapter(_FakeSession())

    assert adapter.variant_ids() == ("v0-auto", "v1-auto")
    assert adapter.variant_ids("launch", compilation_id="c0") == (
        "v0-auto",
        "v0-one",
    )
    assert adapter.variant_ids("full") == ("v0-auto", "v0-one", "v1-auto")
    assert adapter.search_space() == {
        "forge_variant": ("choice", ("v0-auto", "v1-auto"))
    }
    assert adapter.bind({"forge_variant": "v0-one"}) == ("bound", "v0-one")

    provider = adapter.ptxas_search_space(version="13.3", variant="sm89")
    assert provider.kwargs == {
        "version": "13.3",
        "variant": "sm89",
        "tag": "latest",
    }


def test_compileiq_adapter_validates_samples_and_emits_plain_manifest():
    adapter = _CompileIQVariantAdapter(_FakeSession())

    with pytest.raises(ValueError, match="launch search requires"):
        adapter.variant_ids("launch")
    with pytest.raises(KeyError, match="unknown Forge compilation group"):
        adapter.variant_ids("launch", compilation_id="missing")
    with pytest.raises(KeyError, match="require 'forge_variant'"):
        adapter.select({})
    with pytest.raises(KeyError, match="unknown Forge kernel variant"):
        adapter.select({"forge_variant": "missing"})

    manifest = adapter.manifest()
    assert manifest["schema_version"] == 1
    assert manifest["structural_variant_ids"] == ("v0-auto", "v1-auto")
    assert [item["compilation_id"] for item in manifest["variants"]] == [
        "c0",
        "c0",
        "c1",
    ]


def test_compileiq_adapter_builds_balanced_exhaustive_stage_schedule():
    adapter = _CompileIQVariantAdapter(_FakeSession())

    schedule = adapter.paired_schedule(blocks=4)
    assert len(schedule) == 8
    assert [trial.variant_id for trial in schedule] == [
        "v0-auto",
        "v0-auto",
        "v0-auto",
        "v0-auto",
        "v1-auto",
        "v1-auto",
        "v1-auto",
        "v1-auto",
    ]
    assert [trial.order for trial in schedule[:4]] == [
        ("baseline", "candidate"),
        ("candidate", "baseline"),
        ("baseline", "candidate"),
        ("candidate", "baseline"),
    ]

    launch = adapter.paired_schedule("launch", compilation_id="c0", blocks=2)
    assert [(trial.variant_id, trial.block) for trial in launch] == [
        ("v0-auto", 0),
        ("v0-auto", 1),
        ("v0-one", 0),
        ("v0-one", 1),
    ]


def test_compileiq_adapter_ranks_complete_paired_evidence_fail_closed():
    adapter = _CompileIQVariantAdapter(_FakeSession())
    ranked = adapter.rank_paired(
        {
            "v0-auto": (0.93, 1.01, 0.95, 0.97),
            "v1-auto": (0.98, 0.99, 0.97, 0.96),
        },
        blocks=4,
    )

    assert [item.variant_id for item in ranked] == ["v1-auto", "v0-auto"]
    assert ranked[0].worst_ratio == pytest.approx(0.99)
    assert ranked[0].median_ratio == pytest.approx(0.975)
    assert ranked[0].worst_positive
    assert ranked[1].worst_ratio == pytest.approx(1.01)
    assert not ranked[1].worst_positive


@pytest.mark.parametrize("blocks", [0, 1, 3, True])
def test_compileiq_adapter_rejects_unbalanced_pairing(blocks):
    adapter = _CompileIQVariantAdapter(_FakeSession())
    with pytest.raises((TypeError, ValueError)):
        adapter.paired_schedule(blocks=blocks)


def test_compileiq_adapter_rejects_incomplete_or_invalid_evidence():
    adapter = _CompileIQVariantAdapter(_FakeSession())
    with pytest.raises(ValueError, match="missing"):
        adapter.rank_paired({"v0-auto": (0.9, 0.9)}, blocks=2)
    with pytest.raises(ValueError, match="exactly 2"):
        adapter.rank_paired({"v0-auto": (0.9,), "v1-auto": (0.9, 0.9)}, blocks=2)
    with pytest.raises(ValueError, match="finite and positive"):
        adapter.rank_paired(
            {"v0-auto": (0.9, float("nan")), "v1-auto": (0.9, 0.9)},
            blocks=2,
        )
