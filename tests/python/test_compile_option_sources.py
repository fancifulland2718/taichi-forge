import taichi_forge as ti
from taichi_forge._lib import core
from taichi_forge.lang import impl


def test_frontend_defaults_and_direct_assignments(monkeypatch):
    for name in ("COMPILE_TIER", "ADVANCED_OPTIMIZATION", "FULL_SIMPLIFY_GLOBAL_ITER_CAP"):
        monkeypatch.delenv("TI_" + name, raising=False)
    ti.init(arch=ti.cpu, _test_mode=True)
    cfg = impl.default_cfg()
    sources = core._get_compile_option_sources(cfg)
    assert sources["compile_tier"] == "frontend_default"
    assert sources["advanced_optimization"] == "derived"
    assert sources["full_simplify_global_iter_cap"] == "unspecified"
    cfg.full_simplify_global_iter_cap = 1
    cfg.advanced_optimization = False
    cfg.llvm_opt_level = 0
    sources = core._get_compile_option_sources(cfg)
    for name in ("full_simplify_global_iter_cap", "advanced_optimization", "llvm_opt_level"):
        assert sources[name] == "direct_assignment"
    ti.reset()


def test_keyword_environment_precedence_preserves_false_and_zero(monkeypatch):
    monkeypatch.setenv("TI_COMPILE_TIER", "full")
    monkeypatch.setenv("TI_ADVANCED_OPTIMIZATION", "1")
    monkeypatch.setenv("TI_LLVM_OPT_LEVEL", "0")
    monkeypatch.setenv("TI_FULL_SIMPLIFY_GLOBAL_ITER_CAP", "1")
    ti.init(arch=ti.cpu, advanced_optimization=False, _test_mode=True)
    cfg = impl.default_cfg()
    sources = core._get_compile_option_sources(cfg)
    assert cfg.compile_tier == "full"
    assert cfg.advanced_optimization is False
    assert cfg.llvm_opt_level == 0
    assert cfg.full_simplify_global_iter_cap == 1
    assert sources["advanced_optimization"] == "init_keyword"
    for name in ("compile_tier", "llvm_opt_level", "full_simplify_global_iter_cap"):
        assert sources[name] == "environment"
    ti.reset()
