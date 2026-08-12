import ast

import taichi_forge as ti
from tests import test_utils

_ARCHES = [ti.cpu, ti.cuda, ti.vulkan]
_EXCLUDE = [(ti.vulkan, "Darwin")]


@test_utils.test(arch=_ARCHES, exclude=_EXCLUDE)
def test_parallel_compile_precompiles_without_launching():
    x = ti.field(ti.i32, shape=())

    @ti.kernel
    def set_x(v: ti.i32):
        x[None] = v

    assert ti.parallel_compile([(set_x, (7,))]) == 1
    assert x[None] == 0

    set_x(7)
    assert x[None] == 7


@test_utils.test(arch=_ARCHES, exclude=_EXCLUDE)
def test_parallel_compile_accepts_kwargs_task():
    x = ti.field(ti.i32, shape=())

    @ti.kernel
    def set_x(a: ti.i32, b: ti.i32):
        x[None] = a * 10 + b

    assert ti.compile_kernels([(set_x, (), {"b": 2, "a": 3})]) == 1
    assert x[None] == 0

    set_x(3, 2)
    assert x[None] == 32


@test_utils.test(arch=_ARCHES, exclude=_EXCLUDE)
def test_parallel_compile_accepts_kernel_opt_level_overrides():
    x = ti.field(ti.i32, shape=())

    @ti.kernel(opt_level="fast")
    def set_fast(v: ti.i32):
        x[None] = v

    @ti.kernel(opt_level="full")
    def set_full(v: ti.i32):
        x[None] = v * 10

    assert ti.compile_kernels([(set_fast, (2,)), (set_full, (3,))]) == 2
    assert x[None] == 0

    set_fast(2)
    assert x[None] == 2
    set_full(3)
    assert x[None] == 30


@test_utils.test(arch=_ARCHES, exclude=_EXCLUDE)
def test_parallel_compile_accepts_duplicate_specializations():
    x = ti.field(ti.i32, shape=())

    @ti.kernel
    def set_x(v: ti.i32):
        x[None] = v

    assert ti.compile_kernels([(set_x, (4,)), (set_x, (4,)), (set_x, (4,))]) == 3
    assert x[None] == 0

    set_x(4)
    assert x[None] == 4


@test_utils.test(arch=_ARCHES, exclude=_EXCLUDE)
def test_compile_profile_captures_python_frontend_events():
    x = ti.field(ti.i32, shape=())

    @ti.func
    def add_one(v):
        return v + 1

    @ti.kernel
    def set_x(v: ti.i32):
        x[None] = add_one(v)

    with ti.compile_profile() as prof:
        ti.parallel_compile([(set_x, (5,))])

    rows = prof.python_events()
    paths = [row["path"] for row in rows]
    assert any(path.startswith("python.frontend.set_x.") for path in paths)
    assert any(path == "python.kernel.ast_transform:set_x" for path in paths)
    assert any(path == "python.func.inline_transform:add_one" for path in paths)
    assert any(path == "python.parallel_compile.submit" for path in paths)
    assert prof.top_n(5)


@test_utils.test(arch=_ARCHES, exclude=_EXCLUDE, offline_cache=False)
def test_compile_profile_captures_cpp_ir_events():
    x = ti.field(ti.i32, shape=())

    @ti.kernel
    def d5_cpp_profile_set(v: ti.i32):
        x[None] = v * 17 + 1

    with ti.compile_profile() as prof:
        d5_cpp_profile_set(6)
        ti.sync()

    rows = prof.records(include_python=False)
    paths = [row["path"] for row in rows]
    assert any("cpp.compile.ir_pipeline" in path for path in paths)
    assert any("cpp.compile.backend_codegen" in path for path in paths)
    assert any("cpp.ir." in path for path in paths)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_compile_profile_separates_snode_lifecycle_phases():
    value = ti.field(ti.i32)
    builder = ti.FieldsBuilder()
    builder.pointer(ti.i, 8).dense(ti.i, 4).place(value)

    with ti.compile_profile() as prof:
        tree = builder.finalize()
        tree.destroy()

    paths = [row["path"] for row in prof.records(include_python=False)]
    for phase in (
        "cpp.snode.add.total",
        "cpp.snode.add.lifecycle_lock",
        "cpp.snode.add.backend_materialize",
        "cpp.snode.add.layout_fingerprint",
        "cpp.snode.destroy.total",
        "cpp.snode.destroy.lifecycle_lock",
        "cpp.snode.destroy.backend_sync",
        "cpp.snode.destroy.frontend_caches",
        "cpp.snode.destroy.backend_resources",
        "cpp.snode.destroy.executables",
        "cpp.snode.destroy.kernel_definitions",
    ):
        assert any(phase in path for path in paths), phase


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_snode_relocation_manifest_is_machine_generated_and_fail_closed():
    value = ti.field(ti.i32)
    builder = ti.FieldsBuilder()
    builder.pointer(ti.i, 8).dense(ti.i, 4).place(value)
    tree = builder.finalize()

    @ti.kernel
    def sum_active() -> ti.i32:
        total = 0
        for i in value:
            total += value[i]
        return total

    key = sum_active._primal.ensure_compiled()
    kernel_cpp = sum_active._primal.compiled_kernels[key]
    manifest = dict(ti.lang.impl.get_runtime().prog._debug_snode_relocation_manifest(kernel_cpp))
    tree_id = tree.id
    tree_generation = tree.generation
    tree.destroy()

    assert manifest["schema_version"] == 1
    assert manifest["has_snode_tree_dependencies"]
    assert manifest["coverage"] == ("compiled_dependency_and_task_inventory_only")
    assert not manifest["compiler_embedded_state_fully_classified"]
    assert not manifest["reuse_admitted"]
    assert len(manifest["tree_dependencies"]) == 1
    dependency = manifest["tree_dependencies"][0]
    assert dependency["tree_id"] == tree_id
    assert dependency["generation"] == tree_generation
    assert dependency["layout_fingerprint"] != 0
    task_types = {task["task_type"] for task in manifest["tasks"]}
    assert "listgen" in task_types
    assert "struct_for" in task_types
    assert all(not task["embedded_state_audited"] for task in manifest["tasks"])
    assert "sparse_list_and_allocator_relocation_not_qualified" in (manifest["blockers"])


@test_utils.test(arch=ti.cpu)
def test_source_template_cache_reuses_ast_template_for_specializations():
    x = ti.field(ti.i32, shape=())

    @ti.kernel
    def set_x(v: ti.template()):
        if ti.static(v == 3):
            x[None] = 3
        else:
            x[None] = 5

    set_x(3)
    cache = set_x._primal._source_template_cache
    assert len(cache) == 5
    assert isinstance(cache[4], ast.Module)
    template_id = id(cache[4])

    set_x(5)
    assert id(set_x._primal._source_template_cache[4]) == template_id
    assert len(set_x._primal.compiled_kernels) == 2
    assert x[None] == 5


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_kernel_cache_key_separates_compile_tier_changes():
    x = ti.field(ti.i32, shape=())

    @ti.kernel
    def set_x(v: ti.i32):
        x[None] = v

    set_x(1)
    key = set_x._primal.ensure_compiled(2)
    set_x._primal.compiled_kernels[key].set_compile_tier_override("fast")
    with ti.compile_profile() as prof:
        set_x(2)

    paths = [row["path"] for row in prof.records(include_python=False)]
    assert any("cpp.compile.backend_codegen" in path for path in paths)
    assert x[None] == 2
