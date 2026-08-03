#!/usr/bin/env python3
"""Validate an installed taichi-forge runtime/shim wheel pair."""

import faulthandler
import importlib
import os
import platform
import re
import sys
import tempfile
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path


faulthandler.enable(all_threads=True)


def _checkpoint(stage: str) -> None:
    print(f"[installed-runtime] {stage}", file=sys.stderr, flush=True)


_checkpoint("import numpy: start")
np = importlib.import_module("numpy")

_checkpoint("import numpy: passed")

_checkpoint("import taichi_forge: start")
ti = importlib.import_module("taichi_forge")
_checkpoint("import taichi_forge: passed")


def _validate_distribution_versions() -> str:
    shim_version = metadata.version("taichi-forge")
    runtime_version = metadata.version("taichi-forge-runtime")
    if shim_version != runtime_version:
        raise RuntimeError(
            "installed shim/runtime version mismatch: "
            f"taichi-forge={shim_version}, "
            f"taichi-forge-runtime={runtime_version}"
        )
    return shim_version


def _validate_build_identity() -> str:
    commit = ti._lib.core.get_commit_hash()
    if not commit or not re.fullmatch(r"[0-9a-fA-F]{7,40}", commit):
        raise RuntimeError(f"invalid native runtime commit identity: {commit!r}")
    return commit


def _runtime_package_dirs() -> list[Path]:
    spec = find_spec("taichi_forge_runtime")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("taichi_forge_runtime package is not importable")
    return [Path(path).resolve() for path in spec.submodule_search_locations]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_cudart_belongs_to_runtime_package(path: Path) -> None:
    candidate_roots = []
    for package_dir in _runtime_package_dirs():
        candidate_roots.append(package_dir)
        candidate_roots.append(
            package_dir.parent / f"{package_dir.name}.libs"
        )
    resolved = path.resolve()
    if not any(
        root.is_dir() and _is_relative_to(resolved, root.resolve())
        for root in candidate_roots
    ):
        raise RuntimeError(
            "bundled CUDART was not loaded from taichi-forge-runtime: "
            f"{resolved}"
        )


def _packaged_cuda_runtime_major(path: Path) -> int:
    name = path.name.lower()
    if platform.system() == "Windows":
        match = re.fullmatch(r"cudart64_(\d+)\.dll", name)
    elif platform.system() == "Linux":
        match = re.fullmatch(
            r"(?:libcudart\.so\.|libcudart-[^.]+\.so\.)(\d+)(?:\.\d+)*",
            name,
        )
    else:
        raise RuntimeError(
            f"unsupported platform for bundled CUDART validation: {platform.system()}"
        )
    if match is None:
        raise RuntimeError(f"unrecognized bundled CUDART name: {path.name}")
    return int(match.group(1))


def _packaged_cudart_candidates() -> list[Path]:
    candidates = []
    for package_dir in _runtime_package_dirs():
        roots = [package_dir, package_dir.parent / f"{package_dir.name}.libs"]
        for root in roots:
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    _packaged_cuda_runtime_major(path)
                except RuntimeError:
                    continue
                candidates.append(path.resolve())
    return sorted(set(candidates))


def _validate_packaged_cuda_runtime() -> tuple[Path | None, int | None]:
    candidate = os.environ.get("TI_CUDA_CUB_SORT_BUNDLED_CUDART_PATH", "")
    if not candidate:
        stray = _packaged_cudart_candidates()
        if stray:
            raise RuntimeError(
                "installed driver-only runtime contains undiscovered CUDART: "
                f"{stray}"
            )
        return None, None
    path = Path(candidate)
    if not path.is_file():
        raise RuntimeError(f"the discovered bundled CUDART does not exist: {path}")
    _validate_cudart_belongs_to_runtime_package(path)

    major = _packaged_cuda_runtime_major(path)
    declared_major = os.environ.get(
        "TI_CUDA_CUB_SORT_BUNDLED_CUDART_MAJOR", ""
    )
    if declared_major:
        try:
            declared_major_value = int(declared_major)
        except ValueError as exc:
            raise RuntimeError(
                f"invalid bundled CUDART manifest major: {declared_major!r}"
            ) from exc
        if declared_major_value != major:
            raise RuntimeError(
                "bundled CUDART manifest/library mismatch: "
                f"manifest={declared_major_value}, library={path.name}"
            )
    return path, major


def _validate_cpu_native_ad() -> None:
    n = 8
    _checkpoint("cpu native AD: init")
    ti.init(arch=ti.cpu)
    _checkpoint("cpu native AD: allocate")
    x = ti.ndarray(ti.f32, shape=n, needs_grad=True)
    y = ti.ndarray(ti.f32, shape=n, needs_grad=True)
    loss = ti.field(ti.f32, shape=(), needs_grad=True)
    x.from_numpy(np.arange(n, dtype=np.float32))
    _checkpoint("cpu native AD: input upload passed")

    @ti.kernel
    def sum_output(values: ti.types.ndarray(dtype=ti.f32, ndim=1)):
        for i in range(n):
            loss[None] += values[i]

    with ti.ad.Tape(loss):
        ti.algorithms.experimental_transform(
            x, y, scale=2.5, bias=1.0, method="cpu_native"
        )
        sum_output(y)

    _checkpoint("cpu native AD: forward and backward passed")
    np.testing.assert_allclose(x.grad.to_numpy(), np.full(n, 2.5, np.float32))
    _checkpoint("cpu native AD: gradient readback passed")
    ti.reset()
    _checkpoint("cpu native AD: reset passed")


def _validate_cpu_dynamic_workload() -> None:
    """Exercise the 0.6.1 device-owned count and worklist contracts."""

    capacity = 17
    requested = capacity + 3
    _checkpoint("cpu dynamic workload: init")
    ti.init(arch=ti.cpu, offline_cache=False)

    capabilities = ti.graph.dynamic_work_capabilities()
    if capabilities.get("schema_version") != 3:
        raise RuntimeError(
            "installed runtime does not expose dynamic-work schema v3"
        )
    if not capabilities.get("worklist", {}).get("available", False):
        raise RuntimeError("installed runtime does not expose DeviceWorklist")

    @ti.kernel
    def append_items(
        values: ti.types.ndarray(dtype=ti.i32, ndim=1),
        extent_state: ti.types.ndarray(dtype=ti.i32, ndim=1),
        generated: ti.types.ndarray(dtype=ti.i32, ndim=0),
        overflow: ti.types.ndarray(dtype=ti.i32, ndim=0),
        bound: ti.i32,
        count: ti.i32,
    ):
        for i in range(count):
            ti.algorithms.device_worklist_append(
                values,
                extent_state,
                generated,
                overflow,
                bound,
                i * 2 + 1,
            )

    @ti.kernel
    def consume_items(
        values: ti.types.ndarray(dtype=ti.i32, ndim=1),
        extent_state: ti.types.ndarray(dtype=ti.i32, ndim=1),
        output: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in range(capacity):
            if i < ti.device_extent_count(extent_state):
                output[i] = values[i] * 3

    worklist = ti.algorithms.DeviceWorklist(capacity, ti.i32)
    worklist.prepare_next()
    append_items(*worklist.append_arguments(), requested)
    worklist.commit_next()
    snapshot = worklist.snapshot()
    if snapshot.extent.count != capacity or not snapshot.extent.overflow:
        raise RuntimeError(
            "DeviceWorklist did not clamp and report an overflowing producer"
        )

    graph_args = worklist.graph_args("wheel_worklist")
    output_arg = ti.graph.Arg(
        ti.graph.ArgKind.NDARRAY, "wheel_worklist_output", ti.i32, ndim=1
    )
    builder = ti.graph.GraphBuilder()
    handle = builder.dispatch_bounded(
        consume_items,
        graph_args.current_values,
        graph_args.current_extent,
        output_arg,
        extent=graph_args.current_extent,
        capacity=capacity,
    )
    graph = builder.compile()
    output = ti.ndarray(ti.i32, shape=capacity)
    runtime_args = {
        graph_args.current_values.name: worklist.values,
        graph_args.current_extent.name: worklist.extent,
        output_arg.name: output,
    }
    graph.run(runtime_args)

    expected = (np.arange(capacity, dtype=np.int32) * 2 + 1) * 3
    np.testing.assert_array_equal(np.sort(output.to_numpy()), expected)
    report = worklist.execution_report(handle)
    if report.useful_count != capacity or report.executed_count != capacity:
        raise RuntimeError(
            "bounded DeviceWorklist consumer reported inconsistent execution"
        )
    ti.reset()
    _checkpoint("cpu dynamic workload: passed")


def _validate_cpu_field_roundtrips() -> None:
    """Exercise field addressing and readback through an installed wheel."""

    with tempfile.TemporaryDirectory(
        prefix="taichi-forge-wheel-cache-"
    ) as cache_dir:
        for threads, offline_cache in ((1, False), (0, True)):
            mode = f"threads={threads or 'default'}, offline_cache={offline_cache}"
            _checkpoint(f"cpu field roundtrip ({mode}): init")
            init_options = {
                "arch": ti.cpu,
                "offline_cache": offline_cache,
                "offline_cache_file_path": cache_dir,
            }
            if threads:
                init_options["cpu_max_num_threads"] = threads
            ti.init(**init_options)
            _checkpoint(f"cpu field roundtrip ({mode}): init passed")

            scalar32 = ti.field(ti.f32, shape=())
            scalar64 = ti.field(ti.f64, shape=())
            one32 = ti.field(ti.f32, shape=1)
            values32 = ti.field(ti.f32, shape=7)
            values64 = ti.field(ti.f64, shape=7)
            serial64 = ti.field(ti.f64, shape=())
            atomic64 = ti.field(ti.f64, shape=())

            @ti.kernel
            def store_fields():
                scalar32[None] = 3.25
                scalar64[None] = 7.5
                one32[0] = -2.0
                for i in range(7):
                    values32[i] = ti.cast(i, ti.f32) * 0.5 - 1.0
                    values64[i] = ti.cast(i, ti.f64) * 0.25 - 0.5

            @ti.kernel
            def read_fields() -> ti.f64:
                result = scalar64[None] + ti.cast(scalar32[None], ti.f64)
                for i in range(7):
                    result += ti.cast(values32[i], ti.f64) + values64[i]
                return result + ti.cast(one32[0], ti.f64)

            @ti.kernel
            def reduce_f64():
                serial64[None] = 0.0
                atomic64[None] = 0.0
                ti.loop_config(serialize=True)
                for i in range(7):
                    serial64[None] += values64[i]
                for i in range(7):
                    ti.atomic_add(atomic64[None], values64[i])

            store_fields()
            _checkpoint(f"cpu field roundtrip ({mode}): kernel store passed")
            expected32 = np.arange(7, dtype=np.float32) * 0.5 - 1.0
            expected64 = np.arange(7, dtype=np.float64) * 0.25 - 0.5
            np.testing.assert_array_equal(values32.to_numpy(), expected32)
            _checkpoint(f"cpu field roundtrip ({mode}): f32 readback passed")
            np.testing.assert_array_equal(values64.to_numpy(), expected64)
            _checkpoint(f"cpu field roundtrip ({mode}): f64 readback passed")
            assert scalar32[None] == np.float32(3.25)
            assert scalar64[None] == np.float64(7.5)
            assert one32[0] == np.float32(-2.0)
            _checkpoint(f"cpu field roundtrip ({mode}): scalar access passed")
            expected_return = (
                7.5 + 3.25 - 2.0 + expected32.sum() + expected64.sum()
            )
            np.testing.assert_allclose(
                read_fields(), expected_return, rtol=0, atol=1e-12
            )
            _checkpoint(f"cpu field roundtrip ({mode}): kernel read passed")

            replacement32 = np.linspace(-3.0, 3.0, 7, dtype=np.float32)
            replacement64 = np.linspace(-1.5, 1.5, 7, dtype=np.float64)
            values32.from_numpy(replacement32)
            _checkpoint(f"cpu field roundtrip ({mode}): f32 upload passed")
            values64.from_numpy(replacement64)
            _checkpoint(f"cpu field roundtrip ({mode}): f64 upload passed")
            np.testing.assert_array_equal(values32.to_numpy(), replacement32)
            np.testing.assert_array_equal(values64.to_numpy(), replacement64)
            _checkpoint(f"cpu field roundtrip ({mode}): replacement readback passed")

            reduce_f64()
            _checkpoint(f"cpu field roundtrip ({mode}): f64 reductions passed")
            expected_sum = replacement64.sum(dtype=np.float64)
            np.testing.assert_allclose(
                serial64[None], expected_sum, rtol=0, atol=1e-12
            )
            np.testing.assert_allclose(
                atomic64[None], expected_sum, rtol=0, atol=1e-12
            )
            ti.reset()
            _checkpoint(f"cpu field roundtrip ({mode}): reset passed")


def main() -> None:
    _checkpoint("distribution versions: start")
    version = _validate_distribution_versions()
    _checkpoint(f"distribution versions: passed ({version})")
    commit = _validate_build_identity()
    _checkpoint(f"native build identity: passed ({commit})")
    cudart, cudart_major = _validate_packaged_cuda_runtime()
    _checkpoint("packaged CUDA runtime: passed")
    _validate_cpu_field_roundtrips()
    _checkpoint("cpu field roundtrips: passed")
    _validate_cpu_native_ad()
    _checkpoint("cpu native AD: passed")
    _validate_cpu_dynamic_workload()
    _checkpoint("cpu dynamic workload: passed")
    if cudart is None:
        dependency = "driver-only; bundled CUDART=none"
    else:
        dependency = f"legacy bundled CUDART major {cudart_major}: {cudart}"
    print(
        "installed runtime validation passed for "
        f"{version} ({commit[:12]}); field/f64 roundtrips passed; {dependency}"
    )


if __name__ == "__main__":
    main()
