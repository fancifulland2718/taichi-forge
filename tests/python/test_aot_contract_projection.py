"""Deployment tree projection preserves the producer and each loaded instance."""

import taichi_forge as ti
from taichi_forge_runtime.aot import Runtime
from tests import test_utils


@test_utils.test(arch=[ti.cpu, ti.cuda], offline_cache=False)
def test_repeated_export_preserves_compiled_tree_contract(tmp_path):
    unrelated = ti.field(ti.i32, shape=4)
    unrelated.fill(19)
    counter = ti.field(ti.i32)
    builder = ti.FieldsBuilder()
    builder.place(counter)
    tree = builder.finalize()
    assert tree.id > 0  # A deployment-local zero must differ from the producer.

    @ti.kernel
    def assign(value: ti.i32):
        counter[None] = value

    @ti.kernel
    def read() -> ti.i32:
        return counter[None]

    assign(7)
    module = ti.aot.Module()
    module.add_kernel(assign)
    module.add_kernel(read)
    entries = ["assign", "read"]
    assert module._aot_builder._used_snode_tree_ids(entries) == [tree.id]
    for name in ("first", "second"):
        module.save(tmp_path / name)
        assert module._aot_builder._used_snode_tree_ids(entries) == [tree.id]
        assert read() == 7

    arch = "cpu" if ti.cfg.arch == ti.cpu else "cuda"
    with Runtime(arch) as runtime:
        first = runtime.load_module(tmp_path / "first")
        second = runtime.load_module(tmp_path / "second")
        first.kernel("assign")(31)
        second.kernel("assign")(42)
        assert first.kernel("read")() == 31
        assert second.kernel("read")() == 42
        first.close()
        assert second.kernel("read")() == 42
    assert read() == 7
    assign(9)
    assert read() == 9
    assert unrelated[0] == 19
