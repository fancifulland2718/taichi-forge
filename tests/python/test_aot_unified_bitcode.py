import hashlib
import importlib
import json
import subprocess
import sys
import textwrap
import zipfile

import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_unified_preserves_per_kernel_prepare_budget(tmp_path):
    from array import array
    from contextlib import closing
    from taichi_forge_runtime.aot import Runtime, Scalar

    @ti.kernel(opt_level="fast")
    def fast(data: ti.types.ndarray(ti.i32, 1), n: ti.i32):
        for i in range(n):
            data[i] = 2 * i

    @ti.kernel(opt_level="full")
    def full(data: ti.types.ndarray(ti.i32, 1), n: ti.i32):
        for i in range(n):
            data[i] += 1

    @ti.kernel
    def default(data: ti.types.ndarray(ti.i32, 1), n: ti.i32):
        for i in range(n):
            data[i] += 2

    package = ti.aot.Module(artifact_format="unified-bc-library",
                            compile_options={"full_simplify_global_iter_cap": 1})
    package.add_kernel(fast)
    package.add_kernel(full)
    package.add_kernel(default)
    path = tmp_path / "budgets.tcm"
    package.archive(str(path))
    with zipfile.ZipFile(path) as archive:
        budgets = {}
        for entry in json.loads(archive.read("unified_abi.json")):
            signature = json.loads(archive.read(f'signatures/{entry["value"]}.json'))
            budgets[entry["key"]] = (signature.get("compile_tier"), signature["full_simplify_global_iter_cap"])
        assert budgets == {"fast": ("fast", 1), "full": ("full", 1), "default": (None, 1)}
    ti.reset()
    with Runtime("vulkan") as runtime, runtime.load_module(path) as library, closing(runtime.ndarray("i32", (4,))) as output:
        for name in ("fast", "full", "default"):
            with ti.compile_profile() as profile:
                bound = library.kernel(name).bind(output, Scalar("i32", 4))
                with bound.prepare(specialize=True, constant_args=(1,)) as prepared:
                    prepared.submit().wait()
            trace = tmp_path / f"{name}.json"
            assert profile.dump_chrome_trace(trace)
            stages = {event["name"] for event in json.loads(trace.read_bytes())}
            assert ("cpp.compile.spirv.optimize_task" in stages) == (name != "fast")
        assert array("i", output.read()).tolist() == [3, 5, 7, 9]


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_unified_native_sparse_iteration_and_instance_isolation(tmp_path):
    from array import array
    from contextlib import closing
    from taichi_forge_runtime.aot import Runtime, Scalar

    fields = ti.FieldsBuilder()
    blocks = fields.pointer(ti.ij, 4, vk_max_active=8)
    cells = blocks.bitmasked(ti.ij, 4)
    data = ti.field(ti.i32)
    cells.place(data)
    fields.finalize()

    @ti.kernel
    def store(x: ti.i32, y: ti.i32, value: ti.i32):
        data[x, y] = value

    @ti.kernel
    def visit(result: ti.types.ndarray(ti.i32, 1)):
        for i, j in data:
            result[0] += 1
            result[1] += (17 * i + j) * data[i, j]

    @ti.kernel
    def remove(x: ti.i32, y: ti.i32):
        ti.deactivate(cells, [x, y])

    package = ti.aot.Module(artifact_format="unified-bc-library")
    slot = package._aot_builder._declare_unified_storage(
        fields.root.ptr, [(blocks.ptr, axis, "blocks") for axis in (0, 1)], "grid")
    # Two capture witnesses for the same logical allocation share exactly one
    # binding even when the concrete samples differ on a model axis.
    witness = ti.FieldsBuilder()
    outer = witness.pointer(ti.ij, 6, vk_max_active=8)
    outer.bitmasked(ti.ij, 4).place(ti.field(ti.i32))
    witness.finalize()
    assert package._aot_builder._declare_unified_storage(
        witness.root.ptr, [(outer.ptr, axis, "blocks") for axis in (0, 1)], "grid") == slot
    for kernel in (store, visit, remove):
        package.add_kernel(kernel)
    path = tmp_path / "active.tcm"
    package.archive(str(path))
    arch = ti.lang.impl.current_cfg().arch.name
    ti.reset()

    def scalar(value):
        return Scalar("i32", value)

    with Runtime(arch) as runtime, runtime.load_module(path) as library:
        with library._bind_native_storage({slot: {"blocks": 3}}) as first, library._bind_native_storage(
            {slot: {"blocks": 6}}
        ) as second, closing(
            runtime.ndarray("i32", (2,))
        ) as output:
            def active(module):
                output.write(bytes(8))
                module.kernel("visit")(output)
                return array("i", output.read()).tolist()

            assert active(first) == [0, 0]
            first.kernel("store")(scalar(1), scalar(2), scalar(3))
            second.kernel("store")(scalar(0), scalar(20), scalar(7))
            assert active(first) == [1, 57]
            assert active(second) == [1, 140]
            first.kernel("remove")(scalar(1), scalar(2))
            assert active(first) == [0, 0]
            assert active(second) == [1, 140]


@test_utils.test(arch=ti.cuda, offline_cache=False)
def test_standalone_unified_compiler_uses_existing_cuda_runtime(tmp_path):
    from array import array
    from contextlib import closing
    from taichi_forge_runtime.aot import Runtime

    @ti.kernel
    def fill(data: ti.types.ndarray(ti.i32, 1)):
        for i in range(data.shape[0]):
            data[i] = data.shape[0] + i

    producer = ti.aot.Module(artifact_format="unified-bc-library")
    producer.add_kernel(fill)
    path = tmp_path / "fill.tcm"
    producer.archive(str(path))
    ti.reset()

    with Runtime("cuda") as runtime, runtime.load_module(path) as library:
        with ti.compile_profile() as profile:
            kernel = library.kernel("fill")
        stages = {row["path"].split("/")[-1] for row in profile.records(include_python=False)}
        assert "cpp.unified.prepare.read_kernel" in stages
        assert "cuda_driver_module_load" not in stages
        with closing(runtime.ndarray("i32", (4,))) as data:
            with kernel.bind(data).prepare(specialize=True) as call:
                call.submit().wait()
            assert array("i", data.read()).tolist() == [4, 5, 6, 7]


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_unified_cpu_reuses_varying_dispatch_extents(tmp_path):
    from array import array
    from contextlib import closing

    @ti.kernel
    def dispatch_only(data: ti.types.ndarray(ti.i32, 1)):
        for i in range(data.shape[0]):
            data[i] = i + 1

    @ti.kernel
    def also_used_in_body(data: ti.types.ndarray(ti.i32, 1)):
        for i in range(data.shape[0]):
            data[i] = i + data.shape[0]

    producer = ti.aot.Module(artifact_format="unified-bc-library")
    producer.add_kernel(dispatch_only)
    producer.add_kernel(also_used_in_body)
    path = tmp_path / "dispatch.tcm"
    producer.archive(str(path))

    with ti.aot.borrow_runtime() as runtime, runtime.load_module(path) as module:
        for name in ("dispatch_only", "also_used_in_body"):
            kernel = module.kernel(name)
            for size in (4, 11, 23, 4):
                with closing(runtime.ndarray("i32", (size,))) as data:
                    bound = kernel.bind(data)
                    with ti.compile_profile() as profile, bound.prepare(specialize=True) as call:
                        call.submit().wait()
                    offset = 1 if name == "dispatch_only" else size
                    assert array("i", data.read()).tolist() == [i + offset for i in range(size)]
                    if size == 23:
                        materialized = any("cpp.unified.prepare.compile" in row["path"]
                                           for row in profile.records(include_python=False))
                        assert materialized == (name == "also_used_in_body")


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_unified_vulkan_reuses_specialization_in_new_process(tmp_path, monkeypatch):
    import pytest
    from taichi_forge_runtime.aot._unified_library import select_manifest

    source = tmp_path / "cached_portable.py"
    source.write_text("""import taichi_forge as ti
@ti.kernel
def fill(data: ti.types.ndarray(ti.i32, 1), bias: ti.i32):
    for i in range(data.shape[0]):
        data[i] = i + bias + data.shape[0]
""", encoding="utf8")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(source.stem)
    output = tmp_path / "library"
    try:
        document = ti.aot.export_package(module, output, arch=ti.vulkan, artifact_format="unified-bc-library")
    finally:
        sys.modules.pop(source.stem, None)
    symbol, = select_manifest(document, "vulkan")["entries"]
    consumer = textwrap.dedent("""
        from array import array
        from pathlib import Path
        import json, sys
        import taichi_forge as ti
        from taichi_forge._lib import core
        from taichi_forge_runtime.aot import load_package

        directory, symbol, cache, size, bias, enabled = sys.argv[1:]
        size, bias = int(size), int(bias)
        ti.init(arch=ti.vulkan, enable_fallback=False, offline_cache=enabled == "1",
                offline_cache_file_path=cache)
        with ti.aot.borrow_runtime() as runtime, load_package(runtime, directory) as package:
            data = runtime.ndarray("i32", (size,))
            bound = package._module.kernel(symbol).bind(data, bias)
            frontend = ti.lang.impl.get_runtime()
            frontend.set_kernel_executable_lifecycle_telemetry_enabled(True)
            frontend.debug_kernel_executable_lifecycle_stats(True)
            core.set_compile_profile_runtime_enabled(True)
            with bound.prepare(specialize=True) as prepared:
                prepared.submit().wait()
            core.set_compile_profile_runtime_enabled(False)
            trace = Path(cache).parent / "prepare-trace.json"
            assert core.export_compile_profile_trace(str(trace))
            restores = sum(e["name"] == "cpp.unified.prepare.read_kernel"
                           for e in json.loads(trace.read_bytes()))
            stats = frontend.debug_kernel_executable_lifecycle_stats()
            assert array("i", data.read()).tolist() == [i + bias + size for i in range(size)]
            assert restores == stats["compiler_invocations"], (restores, stats)
            print("REUSE=" + json.dumps([stats["compiler_invocations"], stats["disk_loads"]]))
    """)
    cache = tmp_path / "cache"

    def consume(size, bias, enabled, expected):
        result = subprocess.run(
            [sys.executable, "-c", consumer, str(output), symbol, str(cache),
             str(size), str(bias), str(enabled)], capture_output=True, text=True, timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        record, = (line[6:] for line in result.stdout.splitlines() if line.startswith("REUSE="))
        assert json.loads(record) == expected

    # A new allocation and ordinary scalar can reuse the shader. Recover a
    # truncated local cache object by compiling the intact TCM again.
    consume(4, 2, 1, [1, 0])
    consume(4, -7, 1, [0, 1])
    artifact, = cache.rglob("*.tic")
    artifact.write_bytes(b"incomplete")
    consume(4, 6, 1, [1, 0])
    consume(4, 3, 1, [0, 1])
    consume(9, 3, 1, [1, 0])
    consume(4, 8, 0, [1, 0])

    # A cached device shader cannot supply a required unit missing from a newly loaded archive.
    damaged = tmp_path / "missing-root.tcm"
    with zipfile.ZipFile(output / "code.tcm") as archive, zipfile.ZipFile(damaged, "w") as copy:
        root = json.loads(archive.read("unified_llvm.json"))["entries"][symbol]
        for member in archive.infolist():
            if member.filename != f"code_units/{root}.bc":
                copy.writestr(member, archive.read(member))
    ti.reset()
    ti.init(arch=ti.vulkan, enable_fallback=False, offline_cache=True, offline_cache_file_path=str(cache))
    with ti.aot.borrow_runtime() as runtime, runtime.load_module(damaged) as loaded:
        with pytest.raises(RuntimeError, match="Missing unified LLVM library member"):
            loaded.kernel(symbol)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_unified_shader_loop_local_and_carried_values(tmp_path, monkeypatch):
    import numpy as np
    from taichi_forge_runtime.aot import load_package

    source = tmp_path / "source"
    source.mkdir()
    (source / "loop_slots.py").write_text("""import taichi_forge as ti
@ti.kernel
def update(data: ti.types.ndarray(ti.i32, 1), steps: ti.i32):
    for i in range(data.shape[0]):
        value = data[i]
        previous = 3
        for j in range(steps):
            term = i + j
            if (i + j) % 2 == 0:
                term = value + j
                previous = term - i
            value += term + previous
        data[i] = value
""", encoding="utf8")
    monkeypatch.syspath_prepend(str(source))
    module = importlib.import_module("loop_slots")
    output = tmp_path / "library"
    try:
        ti.aot.export_package(module, output, arch=ti.vulkan, artifact_format="unified-bc-library")
        with ti.aot.borrow_runtime() as runtime, load_package(runtime, output) as package:
            data = runtime.ndarray("i32", (9,))
            for steps in (0, 1, 5):
                expected = np.arange(-4, 5, dtype=np.int32)
                data.write(expected.tobytes())
                for i, value in enumerate(expected):
                    previous = 3
                    for j in range(steps):
                        term = i + j
                        if (i + j) % 2 == 0:
                            term = value + j
                            previous = term - i
                        value += term + previous
                    expected[i] = value
                package.host.update(data, steps)
                np.testing.assert_array_equal(np.frombuffer(data.read(), np.int32), expected)
    finally:
        sys.modules.pop(module.__name__, None)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_unified_preserves_quarter_turn_rounding_for_prepare(tmp_path, monkeypatch):
    import numpy as np
    from taichi_forge_runtime.aot import load_package

    src = tmp_path / "source"
    src.mkdir()
    path = src / "portable_rotation.py"
    path.write_text("""import taichi_forge as ti
@ti.func
def normalized(q):
    length = ti.sqrt(q.dot(q))
    unit = ti.Vector([1., 0., 0., 0.])
    if length >= 1e-8:
        unit = q / length
    return unit

@ti.kernel
def rotate(quaternions: ti.types.ndarray(ti.f32, 2), matrices: ti.types.ndarray(ti.f32, 3)):
    for i in range(quaternions.shape[0]):
        q = normalized(ti.Vector([quaternions[i, j] for j in ti.static(range(4))]))
        w, x, y, z = q
        rotation = ti.Matrix([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
        ])
        for row, col in ti.static(ti.ndrange(3, 3)):
            matrices[i, row, col] = rotation[row, col]
""", encoding="utf8")
    monkeypatch.syspath_prepend(str(src))
    module = importlib.import_module(path.stem)
    output = tmp_path / "library"
    try:
        ti.aot.export_package(module, output, arch=[ti.vulkan], artifact_format="unified-bc-library")
        half = np.float32(np.sqrt(0.5))
        rotations = np.array([[half, 0, 0, half], [half, 0, 0, -half]], dtype=np.float32)
        source = ti.ndarray(ti.f32, rotations.shape)
        source.from_numpy(rotations)
        dest = ti.ndarray(ti.f32, (2, 3, 3))
        module.rotate(source, dest)
        expected = dest.to_numpy()
        with ti.aot.borrow_runtime() as runtime, load_package(runtime, output) as package:
            data = runtime.ndarray("f32", rotations.shape)
            data.write(rotations.tobytes())
            result = runtime.ndarray("f32", expected.shape)
            package.host.rotate(data, result)
            actual = np.frombuffer(result.read(), np.float32).reshape(expected.shape)
            # Near zero, the change in contraction caused by producer-side
            # expression sharing was large in ULPs despite a tiny absolute delta.
            np.testing.assert_array_max_ulp(actual, expected, maxulp=1)
    finally:
        sys.modules.pop(path.stem, None)


@test_utils.test(arch=ti.vulkan, offline_cache=False)
def test_unified_sampled_image_keeps_numeric_options_for_prepare(tmp_path, monkeypatch):
    import numpy as np
    from taichi_forge_runtime.aot import load_package
    from taichi_forge_runtime.aot._unified_library import select_manifest

    src = tmp_path / "src"
    src.mkdir()
    monkeypatch.syspath_prepend(str(src))
    (src / "portable_sampler.py").write_text("""import taichi_forge as ti
@ti.data_oriented
class Sampler:
    def __init__(self, bright):
        self.bright = bright
        scope = ti.experimental.ResourceScope(self)
        scope.static("bright", choices=(False, True), kernel_only=True)
    @ti.kernel
    def read(self, image: ti.types.texture(2), out: ti.types.ndarray(ti.f32, 1)):
        for i in range(4):
            value = image.fetch(ti.Vector([0, 0]), 0).x
            if ti.static(self.bright):
                value *= 2.0
            out[i] = value
""", encoding="utf8")
    output = tmp_path / "library"
    document = ti.aot.export_package(importlib.import_module("portable_sampler"), output,
                                     arch=["cpu", "cuda", "vulkan"], source_root=src,
                                     artifact_format="unified-bc-library")
    manifest = select_manifest(document, "vulkan")
    assert len(manifest["entries"]) == 1
    entry = next(iter(manifest["entries"].values()))
    assert any(capture.get("prepare_static") for capture in entry["captures"])
    with zipfile.ZipFile(output / "code.tcm") as archive:
        assert len(json.loads(archive.read("unified_llvm.json"))["entries"]) == 1
    with ti.aot.borrow_runtime() as runtime, load_package(runtime, output) as package:
        pixels = runtime.ndarray("f32", (2, 2))
        pixels.write(np.full((2, 2), 3, np.float32))
        image = runtime.native_resource("image", "sampled", parameters=dict(shape=(2, 2), format="r32f",
                                        filter="nearest"), bindings=dict(pixels=pixels))
        result = runtime.ndarray("f32", (4,))
        for bright in (False, True):
            sampler = package.host.Sampler(bright)
            for _ in range(2):
                sampler.read(image, result)
                np.testing.assert_array_equal(np.frombuffer(result.read(), np.float32), [6 if bright else 3] * 4)


def test_unified_incremental_capture_preserves_helper_closures(tmp_path, monkeypatch):
    import pytest
    from taichi_forge_runtime.aot import load_package

    targets = [arch for arch in (ti.cpu, ti.cuda, ti.vulkan) if arch in test_utils.expected_archs()]
    if not targets:
        pytest.skip("No unified LLVM frontend selected")
    folder = tmp_path / "source"
    folder.mkdir()
    monkeypatch.syspath_prepend(str(folder))
    path = folder / "unified_cache_inputs.py"
    original = """import taichi_forge as ti
@ti.real_func
def scale(value: ti.i32) -> ti.i32:
    return 2 * value
@ti.real_func
def forward(value: ti.i32) -> ti.i32:
    return scale(value)
@ti.kernel
def apply(x: ti.i32) -> ti.i32:
    return forward(x)
@ti.kernel
def vector(x: ti.types.vector(2, ti.i32)) -> ti.i32:
    return x[0] - x[1]
"""
    cache = tmp_path / "cache"
    caller_edit = original.replace("return forward(x)", "return forward(x) + 3")
    helper_edit = caller_edit.replace("2 * value", "11 * value")
    for name, text, expected in (("cold", original, 14), ("warm", original, 14),
                                 ("caller", caller_edit, 17), ("edited", helper_edit, 80),
                                 ("repaired", helper_edit, 80), ("fresh", helper_edit, 80)):
        ti.reset()
        if name == "fresh":
            cache = tmp_path / "fresh-cache"
        if name == "repaired":
            baseline = tmp_path / "edited/code.tcm"
            with zipfile.ZipFile(baseline) as source, zipfile.ZipFile(tmp_path / "incomplete.tcm", "w") as target:
                roots = set(json.loads(source.read("unified_llvm.json"))["entries"].values())
                for item in source.infolist():
                    if not item.filename.startswith("code_units/") or item.filename[11:-3] in roots:
                        target.writestr(item, source.read(item))
            (tmp_path / "incomplete.tcm").replace(baseline)
        path.write_text(text, encoding="utf8")
        module = importlib.import_module(path.stem)
        try:
            output = tmp_path / name
            ti.aot.export_package(module, output, arch=targets, artifact_format="unified-bc-library",
                                  build_cache=cache)
        finally:
            sys.modules.pop(path.stem, None)
        record = json.loads((cache / "unified/latest.json").read_bytes())
        assert (cache / "unified" / (record["helpers"] + ".tcb")).is_file()
        assert record["produce"]["unified_capture_restores"] == (
            0 if name in ("cold", "fresh") else 2 if name == "warm" else 1)
        assert record["produce"]["unified_archive_reuses"] == int(name == "warm")
        with zipfile.ZipFile(output / "code.tcm") as archive:
            unit_count = sum(key.startswith("code_units/") for key in archive.namelist())
        referenced = record["produce"]["unified_referenced_units"]
        if name == "warm":
            assert referenced == unit_count
        elif name in ("caller", "edited"):
            assert 0 < referenced < unit_count
        else:
            assert referenced == 0
        for target in targets:
            ti.init(arch=target, enable_fallback=False)
            with ti.aot.borrow_runtime() as runtime, load_package(runtime, output) as pkg:
                assert pkg.host.apply(7) == expected
                assert pkg.host.vector([9, 4]) == 5
        if name == "warm":
            assert (tmp_path / "cold/code.tcm").read_bytes() == (output / "code.tcm").read_bytes()
            assert (tmp_path / "cold/package.json").read_bytes() == (output / "package.json").read_bytes()
        if name == "fresh":
            assert (tmp_path / "repaired/code.tcm").read_bytes() == (output / "code.tcm").read_bytes()
            # The partially relinked archive has since been deliberately damaged.
            assert record["tcm_sha256"] == edited_digest
            assert (tmp_path / "repaired/package.json").read_bytes() == (output / "package.json").read_bytes()
        if name == "edited":
            edited_digest = record["tcm_sha256"]


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_unified_incremental_restores_model_workspaces(tmp_path, monkeypatch):
    import numpy as np
    from taichi_forge_runtime.aot import load_package
    from tests.python.test_aot_package import MODEL_LOCAL_TENSOR_SOURCE

    path = tmp_path / "workspace_model.py"
    path.write_text(MODEL_LOCAL_TENSOR_SOURCE, encoding="utf8")
    monkeypatch.syspath_prepend(str(tmp_path))
    model = importlib.import_module(path.stem)
    cache = tmp_path / "cache"
    try:
        for label in ("cold", "warm"):
            output = tmp_path / label
            ti.aot.export_package(model, output, arch=[ti.cpu], artifact_format="unified-bc-library", build_cache=cache)
            with ti.aot.borrow_runtime() as runtime, load_package(runtime, output) as package:
                for n, dof in ((7, 3), (9, 11)):
                    instance = package.host.Model(n, dof)
                    instance.step(2.0)
                    expected = np.where(np.arange(n) % 2 == 1, (np.arange(n) + 1) * 5 * (dof + 1), 0)
                    np.testing.assert_array_equal(instance.result.to_numpy(), expected)
        stats = json.loads((cache / "unified/latest.json").read_bytes())["produce"]
        with zipfile.ZipFile(tmp_path / "warm/code.tcm") as archive:
            assert stats["unified_capture_restores"] == len(json.loads(archive.read("unified_llvm.json"))["entries"])
        assert stats["unified_root_emissions"] == 0
        assert (tmp_path / "cold/code.tcm").read_bytes() == (tmp_path / "warm/code.tcm").read_bytes()
        assert (tmp_path / "cold/package.json").read_bytes() == (tmp_path / "warm/package.json").read_bytes()
    finally:
        sys.modules.pop(path.stem, None)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_unified_incremental_capture_tracks_owner_fields_per_entry(tmp_path, monkeypatch):
    from taichi_forge_runtime.aot import load_package

    source = tmp_path / "direct_fields.py"
    monkeypatch.syspath_prepend(str(tmp_path))
    original = """import taichi_forge as ti
@ti.data_oriented
class Model:
    def __init__(self):
        scope = ti.experimental.ResourceScope(self)
        scope.cells = scope.field(ti.i32, shape=1, initialization='zero')
    @ti.func
    def write_cell(self, x):
        self.cells[0] = x
        return ti.cast(self.cells[0], ti.i32)
    @ti.kernel
    def write(self, x: ti.i32) -> ti.i32:
        return self.write_cell(x)
    @ti.kernel
    def scalar(self, x: ti.i32) -> ti.i32:
        return x + 1
"""
    added = original.replace("    @ti.func", """        builder = ti.FieldsBuilder()
        scratch = ti.experimental.ResourceScope(self, builder=builder)
        scratch.temp = scratch.field(ti.f32, shape=1)
        builder.finalize()
    @ti.func""")
    colocated = original.replace("    @ti.func", "        scope.temp = scope.field(ti.f32, shape=1)\n    @ti.func")
    edited = colocated.replace("scope.field(ti.i32", "scope.field(ti.i16")
    reader = """
@ti.data_oriented
class Reader:
    def __init__(self, model: Model):
        scope = ti.experimental.ResourceScope(self)
        scope.owner_reference('model', model)
    @ti.kernel
    def nested_write(self, x: ti.i32) -> ti.i32:
        return self.model.write_cell(x)
    @ti.kernel
    def double(self, x: ti.i32) -> ti.i32:
        return x * 2
"""
    for label, text, hits, expected in (("seed", original, 0, 65535), ("recovered", original, None, 65535),
                                       ("added", added, 4, 65535),
                                       ("colocated", colocated, 4, 65535),
                                       ("cold-colocated", colocated, 0, 65535),
                                       ("edited", edited, 2, -1), ("cold", edited, 0, -1)):
        source.write_text(text + reader, encoding="utf8")
        module = importlib.import_module(source.stem)
        output = tmp_path / label
        cache = tmp_path / (label + "-cache" if label.startswith("cold") else "cache")
        if label == "recovered":
            history = cache / "portable/declarations/latest.json"
            previous = json.loads(history.read_bytes())
            assert previous["layouts"]
            previous["layouts"] = dict.fromkeys(previous["layouts"], "damaged recipe")
            history.write_text(json.dumps(previous), encoding="utf8")
        try:
            ti.aot.export_package(module, output, arch=[ti.cpu], artifact_format="unified-bc-library",
                                  build_cache=cache)
        finally:
            sys.modules.pop(source.stem)
        stats = json.loads((cache / "unified/latest.json").read_bytes())["produce"]
        if label == "recovered":
            assert 2 <= stats["unified_capture_restores"] < 4
            assert "damaged recipe" not in json.loads(history.read_bytes())["layouts"].values()
        else:
            assert stats["unified_capture_restores"] == hits
        with ti.aot.borrow_runtime() as runtime, load_package(runtime, output) as package:
            model = package.host.Model()
            assert model.write(65535) == expected
            assert model.scalar(3) == 4
            view = package.host.Reader(model)
            assert view.nested_write(65535) == expected
            assert view.double(3) == 6
    for incremental, clean in (("seed", "recovered"), ("colocated", "cold-colocated"), ("edited", "cold")):
        for name in ("code.tcm", "package.json"):
            assert (tmp_path / incremental / name).read_bytes() == (tmp_path / clean / name).read_bytes()


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_unified_incremental_recompiles_after_cell_stride_changes(tmp_path, monkeypatch):
    from taichi_forge_runtime.aot import load_package

    source = tmp_path / "cell_stride.py"
    monkeypatch.syspath_prepend(str(tmp_path))
    original = """import taichi_forge as ti
@ti.data_oriented
class Grid:
    def __init__(self):
        builder = ti.FieldsBuilder()
        scope = ti.experimental.ResourceScope(self, builder=builder)
        cell = scope.dense(ti.i, 3)
        scope.value = scope.field(ti.i32, shape=3, placement=cell, initialization='zero')
        self.tree = builder.finalize()
    @ti.kernel
    def write(self, x: ti.i32):
        self.value[2] = x
"""
    sibling = "        scope.pad = scope.field(ti.i32, shape=3, placement=cell, initialization='zero')\n"
    interleaved = original.replace("        self.tree", sibling + "        self.tree")
    cache = tmp_path / "cache"
    for label, body, hits in (("packed", original, 0), ("strided", interleaved, 0), ("again", interleaved, 1)):
        source.write_text(body, encoding="utf8")
        module = importlib.import_module(source.stem)
        try:
            ti.aot.export_package(module, tmp_path / label, arch=[ti.cpu], artifact_format="unified-bc-library",
                                  build_cache=cache)
        finally:
            sys.modules.pop(source.stem)
        stats = json.loads((cache / "unified/latest.json").read_bytes())["produce"]
        assert stats["unified_capture_restores"] == hits
        with ti.aot.borrow_runtime() as runtime, load_package(runtime, tmp_path / label) as package:
            grid = package.host.Grid()
            grid.write(73)
            assert grid.value.to_numpy().tolist() == [0, 0, 73]
            if label != "packed":
                assert grid.pad.to_numpy().tolist() == [0, 0, 0]


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_unified_incremental_capture_survives_moving_host_declarations(tmp_path, monkeypatch):
    from taichi_forge_runtime.aot import load_package

    src = tmp_path / "source"
    src.mkdir()
    monkeypatch.syspath_prepend(str(src))
    path = src / "moved_declaration.py"
    original = """import taichi_forge as ti
@ti.data_oriented
class Model:
    def __init__(self, bias):
        self.bias = bias
        self.host_value = 2
        scope = ti.experimental.ResourceScope(self)
        scope.static('bias', choices=(3,), kernel_only=True)
    @ti.kernel
    def apply(self, x: ti.i32) -> ti.i32:
        return x + ti.static(self.bias)
"""
    moved = original.replace("self.host_value = 2", "# Updated Host default\n        self.host_value = 17")
    expanded = moved.replace("choices=(3,)", "choices=(3, 9)")
    cache = tmp_path / "cache"
    for label, source in (("initial", original), ("moved", moved), ("repaired", moved), ("expanded", expanded)):
        if label == "repaired":
            # The individual numerical objects still restore. An invalid
            # previous archive must nevertheless be linked again.
            with zipfile.ZipFile(tmp_path / "moved/code.tcm", "a") as archive:
                archive.writestr("unexpected", b"corrupt link output")
        path.write_text(source, encoding="utf8")
        module = importlib.import_module(path.stem)
        output = tmp_path / label
        try:
            ti.aot.export_package(module, output, arch=[ti.cpu], artifact_format="unified-bc-library",
                                  build_cache=cache)
        finally:
            sys.modules.pop(path.stem, None)
        stats = json.loads((cache / "unified/latest.json").read_bytes())["produce"]
        assert stats["unified_capture_restores"] == int(label in ("moved", "repaired"))
        assert stats["unified_archive_reuses"] == int(label == "moved")
        if label == "repaired":
            assert (output / "code.tcm").read_bytes() == (tmp_path / "initial/code.tcm").read_bytes()
        with ti.aot.borrow_runtime() as runtime, load_package(runtime, output) as package:
            for bias in ((3, 9) if label == "expanded" else (3,)):
                model = package.host.Model(bias)
                assert model.apply(7) == 7 + bias
                assert model.host_value == (2 if label == "initial" else 17)


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_unified_lookup_rejects_changed_helper(tmp_path):
    import pytest
    from taichi_forge_runtime.aot import AotError

    @ti.real_func
    def advance(x: ti.i32) -> ti.i32:
        return x + 1

    @ti.kernel
    def step(x: ti.i32) -> ti.i32:
        return advance(x)

    producer = ti.aot.Module(artifact_format="unified-bc-library")
    producer.add_kernel(step)
    path = tmp_path / "source.tcm"
    producer.archive(str(path))
    damaged = tmp_path / "changed-helper.tcm"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(damaged, "w") as dest:
        root, = json.loads(source.read("unified_llvm.json"))
        root_path = f'code_units/{root["value"]}.bc'
        helper, = (p for p in source.namelist() if p.startswith("code_units/") and p != root_path)
        for member in source.namelist():
            # Valid bitcode and ZIP CRC, but the caller names the original digest.
            data = source.read(root_path if member == helper else member)
            dest.writestr(member, data)

    with ti.aot.borrow_runtime() as runtime, runtime.load_module(damaged) as module:
        with pytest.raises(AotError, match="Changed unified LLVM helper") as error:
            module.kernel("step")
        assert error.value.code == -2  # TI_ERROR_CORRUPTED_DATA


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_unified_prepare_forwards_runtime_arrays_through_helpers(tmp_path):
    import numpy as np
    from taichi_forge_runtime.aot import Scalar

    @ti.real_func
    def advance(x: ti.f32, bias: ti.types.ndarray(ti.f32, 1)) -> ti.f32:
        result = x
        for j in range(2):
            result = result * 0.5 + bias[j]
        return result

    @ti.kernel
    def step(count: ti.i32, data: ti.types.ndarray(ti.f32, 1), bias: ti.types.ndarray(ti.f32, 1)):
        for i in range(count):
            data[i] = advance(advance(data[i], bias), bias)

    producer = ti.aot.Module(artifact_format="unified-bc-library")
    producer.add_kernel(step)
    path = tmp_path / "source.tcm"
    producer.archive(str(path))
    with zipfile.ZipFile(path) as archive:
        unit, = json.loads(archive.read("unified_llvm.json"))
        signature, = json.loads(archive.read("unified_abi.json"))
        assert unit["key"] == signature["key"] == "step"

        def read(digest):
            return archive.read(f"code_units/{digest}.bc")

        for count in (12, 5):
            consumer = ti.aot.Module()
            consumer._aot_builder._add_unified_kernel(
                ti.lang.impl.get_runtime().prog, "step", read(unit["value"]),
                archive.read(f'signatures/{signature["value"]}.json').decode(), {(0,): count}, read)
            consumer.archive(str(tmp_path / f"ready-{count}.tcm"))
    with ti.aot.borrow_runtime() as runtime:
        data, bias = runtime.ndarray("f32", (16,)), runtime.ndarray("f32", (2,))
        for count in (12, 5):
            with runtime.load_module(str(tmp_path / f"ready-{count}.tcm")) as module:
                for offset in (0, 7):
                    expected = np.arange(16, dtype=np.float32)
                    values = np.array([0.5, offset + 1], dtype=np.float32)
                    data.write(expected.tobytes())
                    bias.write(values.tobytes())
                    for _ in range(2):
                        for value in values:
                            expected[:count] = expected[:count] * np.float32(0.5) + value
                    with module.kernel("step").bind(Scalar("i32", count), data, bias).prepare() as prepared:
                        prepared.submit().wait()
                    np.testing.assert_array_equal(np.frombuffer(data.read(), np.float32), expected)


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_unified_library_retains_independent_dense_bindings(tmp_path):
    left = ti.field(ti.i32)
    right = ti.field(ti.i32)
    for field in (left, right):
        fields = ti.FieldsBuilder()
        fields.dense(ti.i, 8).place(field)
        fields.finalize()

    @ti.kernel
    def update(dst: ti.template()):
        for i in range(8):
            dst[i] = i + 7

    module = ti.aot.Module(ti.lang.impl.current_cfg().arch, artifact_format="unified-bc-library")
    for name, field in (("left", left), ("right", right)):
        module.add_kernel(update, template_args={"dst": field}, name=name)
    path = tmp_path / "fields.tcm"
    module.archive(str(path))
    with zipfile.ZipFile(path) as package:
        entries = {row["key"]: row["value"] for row in json.loads(package.read("unified_llvm.json"))}
        assert entries["left"] == entries["right"]
        code = package.read("code_units/" + entries["left"] + ".bc")
        assert code.startswith(b"BC\xc0\xde")
        assert hashlib.sha256(code).hexdigest() == entries["left"]
        storage = json.loads(package.read("unified_storage.json"))
        left_tree, = storage["entries"]["left"]
        right_tree, = storage["entries"]["right"]
        assert left_tree != right_tree
        layout = storage["trees"][left_tree]
        assert layout == storage["trees"][right_tree]
        assert len(storage["schemas"]) == 1
        assert storage["schemas"][layout]["nodes"][1]["extents"][0]["constant"] == 8


def test_unified_device_resource_extents_use_one_kernel(tmp_path, monkeypatch):
    import struct
    import pytest
    from taichi_forge_runtime.aot import load_package

    targets = [arch for arch in (ti.cpu, ti.cuda, ti.vulkan) if arch in test_utils.expected_archs()]
    if not targets:
        pytest.skip("No portable consumer selected")
    path = tmp_path / "unified_device_fields.py"
    path.write_text(
        """import taichi_forge as ti
from typing import Final, NoReturn
HAS_OFFSET: Final[bool] = True

@ti.data_oriented
class Work:
    def __init__(self, cooperative):
        self.cooperative = cooperative
        scope = ti.experimental.ResourceScope(self)
        scope.static("cooperative", choices=(False, True), kernel_only=True)
    @ti.kernel
    def publish(self, data: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in range(32):
            data[i] = i
        if ti.static(self.cooperative):
            ti.loop_config(block_dim=32)
            for i in range(32):
                ti.simt.block.sync()
                data[i] = ti.simt.warp.shfl_sync_i32(ti.u32(0xffffffff), i + 1, 0)

def unavailable(message: str) -> NoReturn:
    raise RuntimeError(message)

@ti.kernel
def exchange(data: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    if ti.static(ti.lang.impl.current_cfg().arch in (ti.cuda, ti.vulkan)):
        ti.loop_config(block_dim=8)
        for i in range(8):
            slots = ti.simt.block.SharedArray((8,), ti.i32)
            slots[i] = i + 1
            ti.simt.block.sync()
            data[i] = slots[(i + 1) % 8]
    else:
        unavailable("exchange requires shared workgroup memory")

@ti.real_func
def device_offset(i: ti.i32) -> ti.i32:
    value = i
    if ti.static(HAS_OFFSET and ti.lang.impl.current_cfg().arch == ti.vulkan):
        value += 19
    else:
        value += 4
    return value
@ti.data_oriented
class Buffer:
    FIRST = 1
    SECOND = 2
    def __init__(self, n, unsigned=False):
        self.n = n
        self.unsigned = unsigned
        scope = ti.experimental.ResourceScope(self)
        for name in ('FIRST', 'SECOND'):
            scope.constant(name)
        scope.static('unsigned', choices=(False, True))
        scope.keys = scope.ndarray(ti.u32 if self.unsigned else ti.f32, shape=scope.dim('n'), initialization='zero')
        if ti.lang.impl.current_cfg().arch == ti.cuda:
            scope.data = scope.field(ti.i32, shape=scope.dim('n'))
            scope.scratch = scope.field(ti.i32, shape=scope.dim('n'))
        else:
            scope.data = scope.field(ti.i32, shape=1)
        scope.backup = scope.field_like(self.data, initialization="zero")
    @ti.kernel
    def update(self, i: ti.i32) -> ti.i32:
        self.data[i] = device_offset(i)
        if ti.static(ti.lang.impl.current_cfg().arch == ti.cuda):
            self.scratch[i] = self.data[i] + 2
            self.data[i] = self.scratch[i]
        for _ in range(1 if ti.static(ti.lang.impl.current_cfg().arch == ti.cuda) else 0):
            self.scratch[i] += 1
        return self.data[i]
    @ti.kernel
    def read_scratch(self, i: ti.i32) -> ti.i32:
        return self.scratch[i]
    @ti.kernel
    def exchange_tail(self, data: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        if ti.static(ti.lang.impl.current_cfg().arch in (ti.cuda, ti.vulkan)):
            ti.loop_config(block_dim=8)
            for i in range(8):
                slots = ti.simt.block.SharedArray(
                    (16 if ti.static(ti.lang.impl.current_cfg().arch == ti.vulkan) else self.n,), ti.i32)
                width = 16 if ti.static(ti.lang.impl.current_cfg().arch == ti.vulkan) else self.n
                slots[width - 8 + i] = width + i
                ti.simt.block.sync()
                data[i] = slots[width - 8 + (i + 1) % 8]
        else:
            unavailable("exchange requires shared workgroup memory")
    @ti.kernel
    def lanes(self, data: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        if ti.static(ti.lang.impl.current_cfg().arch == ti.vulkan):
            ti.loop_config(block_dim=self.n)
            for i in range(32):
                data[i] = ti.simt.block.thread_idx()
        else:
            for i in range(32):
                data[i] = i + 32
""", encoding="utf8")
    monkeypatch.syspath_prepend(str(tmp_path))
    root = importlib.import_module(path.stem)
    output = tmp_path / "package"
    try:
        ti.aot.export_package(root, output, arch=targets, artifact_format="unified-bc-library")
    finally:
        sys.modules.pop(path.stem, None)
    with zipfile.ZipFile(output / "code.tcm") as archive:
        index = json.loads(archive.read("unified_llvm.json"))
        assert len(index["entries"]) == 6
        content = archive.read(index["members"]["__content__"]).decode().splitlines()
        assert content == sorted(content)
        absent = json.loads(archive.read(index["members"]["unified_absent_inputs.json"]))
        assert list(absent) == sorted(absent)
        assert all(list(entries) == sorted(entries) for entries in absent.values())
    for target in targets:
        ti.init(arch=target, enable_fallback=False)
        with ti.aot.borrow_runtime() as runtime, load_package(runtime, output) as package:
            model = package.module(path.stem)
            for n in (3, 8):
                owner = model.Buffer(n, n == 8)
                assert owner.keys.to_numpy().dtype.name == ("uint32" if n == 8 else "float32")
                width = n if target == ti.cuda else 1
                assert tuple(owner.data.shape) == (width,)
                assert tuple(owner.backup.shape) == (width,)
                assert owner.backup._group_owner is not owner.data._group_owner
                assert hasattr(owner, "scratch") == (target == ti.cuda)
                assert owner.update(width - 1) == width - 1 + {ti.cpu: 4, ti.cuda: 6, ti.vulkan: 19}[target]
                if target == ti.cuda:
                    assert owner.read_scratch(width - 1) == width - 1 + 7
                else:
                    with pytest.raises(RuntimeError, match="unavailable resource slot"):
                        owner.read_scratch(0)
            data = runtime.ndarray("i32", (8,))
            lane_data = runtime.ndarray("i32", (32,))
            model.Work(False).publish(lane_data)
            assert struct.unpack("32i", lane_data.read()) == tuple(range(32))
            if target == ti.cuda:
                model.Work(True).publish(lane_data)
                assert struct.unpack("32i", lane_data.read()) == (1,) * 32
            else:
                with pytest.raises(RuntimeError, match="cuda_shfl_sync_i32 requires CUDA"):
                    model.Work(True).publish(lane_data)
            for width in (8, 16):
                model.Buffer(width).lanes(lane_data)
                expected = tuple(i % width if target == ti.vulkan else i + 32 for i in range(32))
                assert struct.unpack("32i", lane_data.read()) == expected
            if target == ti.cpu:
                with pytest.raises(RuntimeError, match="(another device|shared workgroup memory)"):
                    model.exchange(data)
            else:
                model.exchange(data)
                assert struct.unpack("8i", data.read()) == (2, 3, 4, 5, 6, 7, 8, 1)
                for n in (8, 12):
                    model.Buffer(n).exchange_tail(data)
                    width = 16 if target == ti.vulkan else n
                    assert struct.unpack("8i", data.read()) == tuple(width + (i + 1) % 8 for i in range(8))


def test_unified_workgroup_width_and_exchange_bind_at_prepare(tmp_path, monkeypatch):
    import numpy as np
    import pytest
    from taichi_forge_runtime.aot import load_package

    targets = [arch for arch in (ti.cpu, ti.cuda, ti.vulkan) if arch in test_utils.expected_archs()]
    if not targets:
        pytest.skip("No portable consumer selected")
    src = tmp_path / "source"
    src.mkdir()
    path = src / "group_exchange.py"
    path.write_text("""import taichi_forge as ti
@ti.data_oriented
class Exchange:
    def __init__(self, width):
        self.width = width
        scope = ti.experimental.ResourceScope(self)
        scope.dim("width")

    @ti.kernel
    def run(self, data: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        ti.loop_config(block_dim=self.width)
        for i in range(data.shape[0]):
            lane = ti.simt.block.thread_idx()
            slots = ti.simt.block.SharedArray((self.width,), ti.i32)
            old = data[i]
            if lane % 2:
                slots[lane] = old + 1
            else:
                slots[lane] = old
            ti.simt.block.sync()
            live = ti.min(self.width, data.shape[0] - (i - lane))
            neighbor = (lane + 1) % live
            other = slots[neighbor]
            ti.simt.block.sync()
            slots[lane] = old + 2 * other
            ti.simt.block.sync()
            data[i] = slots[neighbor] + old + ti.simt.block.global_thread_idx()
""", encoding="utf8")
    monkeypatch.syspath_prepend(str(src))
    root = importlib.import_module(path.stem)
    output = tmp_path / "library"
    cache = tmp_path / "build-cache"
    try:
        ti.aot.export_package(root, tmp_path / "cold", arch=targets, source_root=src,
                              artifact_format="unified-bc-library", build_cache=cache)
        ti.aot.export_package(root, output, arch=targets, source_root=src,
                              artifact_format="unified-bc-library", build_cache=cache)
    finally:
        sys.modules.pop(path.stem, None)
    stats = json.loads((cache / "unified/latest.json").read_bytes())["produce"]
    assert stats["unified_capture_restores"] == 1
    assert (output / "code.tcm").read_bytes() == (tmp_path / "cold/code.tcm").read_bytes()
    with zipfile.ZipFile(output / "code.tcm") as archive:
        assert len(json.loads(archive.read("unified_llvm.json"))["entries"]) == 1
    for arch in targets:
        ti.init(arch=arch, enable_fallback=False)
        with ti.aot.borrow_runtime() as runtime, load_package(runtime, output) as pkg:
            for width in (1, 8, 16, 32, 8):
                size = 99 if arch == ti.cpu else 96
                data = runtime.ndarray("i32", (size,))
                model = pkg.host.Exchange(width)
                for salt in (7, 23):
                    before = (np.arange(size, dtype=np.int32) + salt) ** 2
                    expected = before.copy()
                    for start in range(0, size, width):
                        part = before[start:start + width]
                        initial = part + np.arange(len(part), dtype=np.int32) % 2
                        expected[start:start + width] = part + np.roll(part + 2 * np.roll(initial, -1), -1)
                    expected += np.arange(size, dtype=np.int32)
                    data.write(before.tobytes())
                    model.run(data)
                    np.testing.assert_array_equal(np.frombuffer(data.read(), np.int32), expected)
                    if arch == ti.cpu or (arch == ti.cuda and width == 16 and salt == 7):
                        jit_data = before.copy()
                        root.Exchange(width).run(jit_data)
                        np.testing.assert_array_equal(jit_data, expected)
            if arch == ti.vulkan:
                from taichi_forge.tools.vulkan_diagnostics import vulkan_runtime_diagnostics

                device = vulkan_runtime_diagnostics()["device"]
                limit = min(device["max_workgroup_invocations"], device["max_workgroup_size_x"])
                for invalid_width in (3, 1 << limit.bit_length()):
                    with pytest.raises(RuntimeError, match="(workgroup size|device limit)"):
                        pkg.host.Exchange(invalid_width).run(data)


def test_unified_cpu_uses_consumer_default_workgroup_width(tmp_path, monkeypatch):
    import numpy as np
    import pytest
    from taichi_forge_runtime.aot import load_package

    if ti.cpu not in test_utils.expected_archs():
        pytest.skip("No CPU consumer selected")
    path = tmp_path / "default_group_width.py"
    path.write_text("""import taichi_forge as ti
@ti.kernel
def lanes(data: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    for i in range(data.shape[0]):
        data[i] = ti.simt.block.thread_idx()
""", encoding="utf8")
    monkeypatch.syspath_prepend(str(tmp_path))
    output = tmp_path / "library"
    try:
        ti.aot.export_package(importlib.import_module(path.stem), output, arch=[ti.cpu],
                              artifact_format="unified-bc-library")
    finally:
        sys.modules.pop(path.stem, None)
    for width in (4, 16):
        ti.init(arch=ti.cpu, default_cpu_block_dim=width, enable_fallback=False)
        with ti.aot.borrow_runtime() as runtime, load_package(runtime, output) as package:
            data = runtime.ndarray("i32", (35,))
            package.host.lanes(data)
            np.testing.assert_array_equal(np.frombuffer(data.read(), np.int32), np.arange(35) % width)


def test_unified_cpu_requires_convergent_group_barriers(tmp_path, monkeypatch):
    import pytest
    from taichi_forge_runtime.aot import load_package

    if ti.cpu not in test_utils.expected_archs():
        pytest.skip("No CPU consumer selected")
    path = tmp_path / "conditional_group.py"
    path.write_text("""import taichi_forge as ti
@ti.kernel
def incompatible(data: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    ti.loop_config(block_dim=8)
    for i in range(data.shape[0]):
        if data[i] > 0:
            ti.simt.block.sync()
        data[i] = 5

@ti.kernel
def device_vote(data: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    ti.loop_config(block_dim=8)
    for i in range(data.shape[0]):
        data[i] = ti.simt.block.sync_count_nonzero(i % 2)

@ti.kernel
def warp_mask(data: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    for i in range(data.shape[0]):
        data[i] = ti.simt.warp.active_mask()
""", encoding="utf8")
    monkeypatch.syspath_prepend(str(tmp_path))
    root = importlib.import_module(path.stem)
    output = tmp_path / "library"
    try:
        ti.aot.export_package(root, output, arch=[ti.cpu], artifact_format="unified-bc-library")
    finally:
        sys.modules.pop(path.stem, None)
    ti.init(arch=ti.cpu)
    with ti.aot.borrow_runtime() as runtime, load_package(runtime, output) as pkg:
        data = runtime.ndarray("i32", (16,))
        with pytest.raises(RuntimeError, match="convergent top-level phases"):
            pkg.host.incompatible(data)
        with pytest.raises(RuntimeError, match="requires CUDA"):
            pkg.host.device_vote(data)
        with pytest.raises(RuntimeError, match="cuda_active_mask requires CUDA"):
            pkg.host.warp_mask(data)


def test_unified_prepare_specializes_device_branches_and_auto_loops(tmp_path, monkeypatch):
    from taichi_forge_runtime.aot._unified_library import select_manifest

    targets = [arch for arch in (ti.cpu, ti.cuda, ti.vulkan) if arch in test_utils.expected_archs()]
    if not targets:
        import pytest

        pytest.skip("No unified LLVM frontend selected")
    source = tmp_path / "unified_source.py"
    source.write_text(
        """import taichi_forge as ti

@ti.kernel
def common(data: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    for i in range(data.shape[0]):
        data[i] += 1

@ti.kernel
def explicit_threads(data: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    ti.loop_config(parallelize=8)
    for i in range(data.shape[0]):
        data[i] += 1

@ti.kernel
def update_second_row(data: ti.types.ndarray(dtype=ti.i32, ndim=2)):
    data[1, 0] += 7

@ti.kernel
def upload_pair(data: ti.types.ndarray(dtype=ti.i32, ndim=1), pair: ti.types.vector(2, ti.i32)):
    data[0] = pair[0]
    data[1] = pair[1]

@ti.kernel
def transpose(src: ti.types.ndarray(dtype=ti.types.matrix(2, 3, ti.f32), ndim=1),
              dst: ti.types.ndarray(dtype=ti.types.matrix(3, 2, ti.f32), ndim=1)):
    for i in range(src.shape[0]):
        dst[i] = src[i].transpose()
        old = ti.atomic_add(dst[i], src[i].transpose())
        dst[i] += old

@ti.kernel
def control(data: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    for i in range(data.shape[0]):
        total = 0
        for j in range(data[i]):
            if j % 2:
                if j == 3:
                    continue
                total += j
            else:
                total -= 2
            total += 5
        k = 0
        while k < data[i]:
            k += 1
            if k == 2:
                total += 7
                break
            total -= 1
        k = 0
        while k < data[i]:
            k += 1
            if k % 2:
                total += 10
            else:
                total += 20
                if k >= 4:
                    break
            total += 1
        data[i] = total

@ti.func
def feature_pair(x):
    if ti.static(ti.lang.impl.current_cfg().arch == ti.vulkan
                 and ti.lang.impl.get_runtime().prog.get_device_caps().get(ti.DeviceCapability.spirv_has_float64)):
        return x + 2, x + 3
    else:
        return x - 2, x - 3

@ti.kernel
def routed(data: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    data[1], data[2] = feature_pair(data[1])
    if ti.static(ti.lang.misc.is_extension_supported(ti.lang.impl.current_cfg().arch, ti.extension.data64)):
        data[3] += 4
    if ti.static(ti.lang.impl.current_cfg().arch == ti.cuda):
        data[0] = 17
    else:
        data[0] = 23

@ti.kernel
def scheduled(data: ti.types.ndarray(dtype=ti.i32, ndim=1)):
    ti.loop_config(block_dim=8)
    if ti.static(ti.lang.impl.current_cfg().arch == ti.cuda):
        ti.loop_config(block_dim=1)
    if ti.static(ti.lang.impl.current_cfg().arch == ti.cpu):
        ti.loop_config(serialize=True)
    for i in range(8):
        if ti.static(ti.lang.impl.current_cfg().arch == ti.vulkan):
            data[i] = ti.simt.block.thread_idx()
        else:
            data[i] = i + 8
""", encoding="utf8")
    monkeypatch.syspath_prepend(str(tmp_path))
    root = importlib.import_module("unified_source")
    try:
        output = tmp_path / "package"
        document = ti.aot.export_package(root, output, artifact_format="unified-bc-library")
        restricted = tmp_path / "restricted"
        restricted_document = ti.aot.export_package(
            root, restricted, arch=targets[0], artifact_format="unified-bc-library")
        assert set(document["targets"]) == {"x64", "cuda", "vulkan"}
        assert restricted_document["targets"] == [targets[0].name]
    finally:
        import sys

        sys.modules.pop(root.__name__, None)
    with zipfile.ZipFile(output / "code.tcm") as archive:
        table = json.loads(archive.read("unified_llvm.json"))
        assert set(table) == {"version", "entries", "members"} and table["version"] == 2
        with zipfile.ZipFile(restricted / "code.tcm") as subset:
            assert json.loads(subset.read("unified_llvm.json"))["entries"] == table["entries"]
            units = {item for item in archive.namelist() if item.startswith("code_units/")}
            assert units == {item for item in subset.namelist() if item.startswith("code_units/")}
            assert all(archive.read(unit) == subset.read(unit) for unit in units)
        functions = {}
        for target in targets:
            manifest = select_manifest(document, target.name)
            functions[target] = {manifest["entries"][name]["definition"].split(".")[-1]: unit
                                 for name, unit in table["entries"].items()}
            assert "unified_abi.json" in table["members"]
        assert len({row["common"] for row in functions.values()}) == 1
        assert len({row["explicit_threads"] for row in functions.values()}) == 1
        assert all(row["common"] != row["explicit_threads"] for row in functions.values())
        assert len({row["routed"] for row in functions.values()}) == 1
        assert len({row["transpose"] for row in functions.values()}) == 1
        assert len(units) == 8
    import numpy as np
    from unittest.mock import patch
    from taichi_forge_runtime.execution import ExecutionSession
    from taichi_forge_runtime.aot import _unified_library

    for target in targets:
        ti.init(arch=target, enable_fallback=False)
        with (
            ti.aot.borrow_runtime() as runtime,
            patch.object(_unified_library, "select_manifest", wraps=select_manifest) as select,
            ExecutionSession(mode="aot", arch=target.name, runtime=runtime, package_paths=[output]) as session,
        ):
            model = importlib.import_module("unified_source")
            pair = runtime.ndarray("i32", (2,))
            for values in ([9, -2], [-5, 11]):
                model.upload_pair(pair, values)
                runtime.wait()
                np.testing.assert_array_equal(np.frombuffer(pair.read(), np.int32), values)
            lanes = runtime.ndarray("i32", (8,))
            model.scheduled(lanes)
            expected_lanes = np.arange(8) + (0 if target == ti.vulkan else 8)
            np.testing.assert_array_equal(np.frombuffer(lanes.read(), np.int32), expected_lanes)
            assert session.loaded_packages == (output,)
            assert sum(call.args[0].get("tcm_sha256") == document["tcm_sha256"]
                       for call in select.call_args_list) == 1
            values = runtime.ndarray("i32", (4,))
            src = runtime.ndarray("f32", (4,), element_shape=(2, 3))
            dst = runtime.ndarray("f32", (4,), element_shape=(3, 2))
            for start in (3, 11):
                expected = np.full(4, start, np.int32)
                values.write(expected.tobytes())
                model.routed(values)
                model.common(values)
                runtime.wait()
                expected[0] = 17 if target == ti.cuda else 23
                has_fp64 = ti.lang.impl.get_runtime().prog.get_device_caps().get(ti.DeviceCapability.spirv_has_float64)
                sign = 1 if target == ti.vulkan and has_fp64 else -1
                expected[1:3] += sign * np.array([2, 3], np.int32)
                if ti.lang.misc.is_extension_supported(target, ti.extension.data64):
                    expected[3] += 4
                np.testing.assert_array_equal(np.frombuffer(values.read(), np.int32), expected + 1)
                matrices = np.arange(24, dtype=np.float32).reshape(4, 2, 3) + start
                src.write(matrices.tobytes())
                model.transpose(src, dst)
                runtime.wait()
                actual = np.frombuffer(dst.read(), np.float32).reshape(4, 3, 2)
                np.testing.assert_array_equal(actual, 3 * matrices.transpose(0, 2, 1))
                counts = np.array([0, 1, 3, 7], dtype=np.int32)
                values.write(counts.tobytes())
                model.control(values)
                runtime.wait()
                expected = [sum(5 + (j if j % 2 else -2) for j in range(n) if j != 3)
                            + (6 if n >= 2 else -n)
                            + sum(11 if k % 2 else 20 + (k < 4)
                                  for k in range(1, min(n, 4) + 1)) for n in counts]
                np.testing.assert_array_equal(np.frombuffer(values.read(), np.int32), expected)
            # A specialized first use can leave the generic executable
            # unmaterialized. Direct callers can still request it later.
            view = select_manifest(document, target.name)
            symbol, = (name for name, header in view["entries"].items()
                       if header["definition"].endswith(".common"))
            module = session._packages[output]._module
            values.write(np.zeros(4, np.int32).tobytes())
            with module.kernel(symbol).bind(values).prepare(specialize=False) as generic:
                generic.submit().wait()
            np.testing.assert_array_equal(np.frombuffer(values.read(), np.int32),
                                          np.ones(4, np.int32))
            if target == ti.vulkan:
                # Equal-shaped slices of one allocation can have different
                # displacements inside an aligned Vulkan descriptor binding.
                from contextlib import ExitStack, closing

                backing = ti.ndarray(ti.i32, 8)
                backing.fill(0)
                with ExitStack() as stack:
                    prepared = []
                    for start in (1, 3):
                        sliced = ti.experimental.ndarray_view(backing, slices=slice(start, start + 3))
                        array = stack.enter_context(closing(runtime.import_dense_view(sliced)))
                        call = module.kernel(symbol).bind(array).prepare(specialize=True)
                        prepared.append(stack.enter_context(call))
                    for call in (prepared[0], prepared[1], prepared[0]):
                        call.submit().wait()
                np.testing.assert_array_equal(backing.to_numpy(), [0, 2, 2, 3, 1, 1, 0, 0])
            if target in (ti.cpu, ti.cuda):
                # The same callable can later receive one member of an
                # interleaved field. Writes must skip the adjacent member.
                from contextlib import closing
                fields = ti.FieldsBuilder()
                member, adjacent = ti.field(ti.i32), ti.field(ti.i32)
                fields.dense(ti.i, 4).place(member, adjacent)
                tree = fields.finalize()
                try:
                    initial = np.arange(4, dtype=np.int32)
                    member.from_numpy(initial)
                    adjacent.fill(71)
                    with closing(runtime.import_dense_view(ti.experimental.ndarray_view(member))) as sliced:
                        with module.kernel(symbol).bind(sliced).prepare(specialize=True) as ready:
                            ready.submit().wait()
                    np.testing.assert_array_equal(member.to_numpy(), initial + 1)
                    np.testing.assert_array_equal(adjacent.to_numpy(), np.full(4, 71))
                finally:
                    tree.destroy()
            # This kernel reads no explicit shape. Only physical addressing
            # depends on the row stride, which differs between these bindings.
            update, = (name for name, header in view["entries"].items()
                       if header["definition"].endswith(".update_second_row"))
            wide = runtime.ndarray("i32", (2, 3))
            narrow = runtime.ndarray("i32", (3, 2))
            zeros = np.zeros(6, np.int32)
            wide.write(zeros.tobytes())
            narrow.write(zeros.tobytes())
            with (module.kernel(update).bind(wide).prepare(specialize=True) as first,
                  module.kernel(update).bind(narrow).prepare(specialize=True) as second):
                first.submit().wait()
                second.submit().wait()
                first.submit().wait()
            np.testing.assert_array_equal(np.frombuffer(wide.read(), np.int32), [0, 0, 0, 14, 0, 0])
            np.testing.assert_array_equal(np.frombuffer(narrow.read(), np.int32), [0, 0, 7, 0, 0, 0])

    if ti.vulkan in targets:
        # A standalone Runtime can expose fewer capabilities than the
        # compiler's default Vulkan device. The helper must see that Runtime.
        import ctypes
        from taichi_forge_runtime.aot import Runtime, load_package
        from taichi_forge_runtime.aot import _c_api as abi

        ti.reset()
        with Runtime("vulkan") as runtime:
            api = runtime._api
            count = abi.UInt()
            api.call(api.ti_get_runtime_capabilities, runtime._handle, ctypes.byref(count), None)
            entries = (abi.CapabilityLevelInfo * count.value)()
            api.call(api.ti_get_runtime_capabilities, runtime._handle, ctypes.byref(count), entries)
            for item in entries:
                if abi.CAPABILITY_NAMES.get(item.capability) == "spirv_has_float64":
                    item.level = 0
            setter = api.library.ti_set_runtime_capabilities_ext
            setter.argtypes = [abi.Handle, abi.UInt, ctypes.POINTER(abi.CapabilityLevelInfo)]
            setter.restype = None
            api.call(setter, runtime._handle, count, entries)
            with load_package(runtime, output) as package:
                data = runtime.ndarray("i32", (4,))
                data.write(np.array([0, 10, 0, 0], np.int32).tobytes())
                package.module("unified_source").routed(data)
                runtime.wait()
                np.testing.assert_array_equal(np.frombuffer(data.read(), np.int32)[:3], [23, 8, 7])
