"""Producer recipes expand once into ordinary kernels and native operations.

No backend implementations live here. The common native registry determines
compiled support; selected helpers use the module's ordinary AOT compiler.
Only the expanded plan/code are deployed, never this registry or Python calls.
"""

from dataclasses import dataclass

from taichi_forge._lib import core as _ti_core


@dataclass(frozen=True)
class _Recipe:
    plan: object
    helpers: tuple = ()


def _sort(name, options):
    from taichi_forge_runtime.aot import ExecutionPlan

    return _Recipe(ExecutionPlan([{"kind": "sort", "keys": "keys", **options}]))


def _compact(name, options):
    from taichi_forge_runtime.aot import ExecutionPlan

    defaults = {"flags": "flags", "count": "count"}
    if "columns" not in options:
        defaults.update(values="values", output="output")
    return _Recipe(ExecutionPlan([{"kind": "compact", **defaults, **options}]))


def _unique(name, options):
    from taichi_forge_runtime.aot import ExecutionPlan

    allowed = {"key_dtype", "keys", "output", "count", "values", "unique_values", "indices"}
    if set(options) - allowed:
        raise ValueError("Unknown unique options: " + ", ".join(sorted(set(options) - allowed)))
    dtype = options.get("key_dtype", "i32")
    if hasattr(dtype, "to_string"):
        dtype = dtype.to_string()
    if dtype not in ("i32", "u32", "i64", "u64"):
        raise ValueError("Unique requires scalar i32/u32/i64/u64 keys")
    if ("values" in options) != ("unique_values" in options):
        raise ValueError("Unique requires both values and unique_values")
    keys, output, count = (options.get(key, key) for key in ("keys", "output", "count"))
    # Resource names are local to the exported recipe, not process identities.
    prefix = "__forge_primitive:" + name + ":"
    flags, extent, starts = (prefix + suffix for suffix in ("heads", "extent", "starts"))
    with_indices = "indices" in options
    entry = "__forge_unique_heads_v1_" + dtype + ("_indices" if with_indices else "")
    arguments = [keys, flags] + ([starts] if with_indices else []) + [extent, extent]
    columns = [{"values": keys, "output": output}]
    resources = [
        {"kind": "ndarray", "name": flags, "dtype": "i32", "shape": [{"array": keys, "axis": 0}]},
        {"kind": "extent", "name": extent, "dtype": "i32", "array": keys, "axis": 0},
    ]
    if "values" in options:
        columns.append({"values": options["values"], "output": options["unique_values"]})
    if with_indices:
        resources.append({"kind": "ndarray", "name": starts, "dtype": "i32", "shape": [{"array": keys, "axis": 0}]})
        columns.append({"values": starts, "output": options["indices"]})
    external = {keys, output, count}
    external.update(options[key] for key in ("values", "unique_values", "indices") if key in options)
    if external.intersection(resource["name"] for resource in resources):
        raise ValueError("Unique binding name collides with a private resource")
    plan = ExecutionPlan(
        [
            {"kind": "kernel", "entry": entry, "arguments": arguments},
            {"kind": "compact", "flags": flags, "count": count, "columns": columns},
        ],
        resources=resources,
    )
    return _Recipe(plan, ((entry, dtype, with_indices),))


# Logical recipes are separate from backend factories: unique is an expansion,
# not a new CUDA/Vulkan algorithm or a consumer-native capability named unique.
_EXPORTS = {
    ("sort", 1): (_sort, (("sort", 1),)),
    ("compact", 1): (_compact, (("compact", 1),)),
    ("unique", 1): (_unique, (("compact", 1),)),
}


def export_primitive(module, name, kind, version, options):
    if not isinstance(name, str) or not name or "\0" in name:
        raise ValueError("Primitive export name must be a nonempty string without NUL")
    if name in module._execution_plans:
        raise ValueError("Duplicate execution plan export: " + name)
    if not isinstance(kind, str) or type(version) is not int or not 0 < version <= 0xFFFFFFFF:
        raise ValueError("Invalid primitive kind or version")
    definition = _EXPORTS.get((kind, version))
    if definition is None:
        raise ValueError(f"Unsupported AOT primitive: {kind} version {version}")
    if "kind" in options:
        raise ValueError("Cannot override the primitive operation kind")
    builder, required = definition
    recipe = builder(name, options)
    for native_kind, native_version in required:
        if not _ti_core.native_operation_capabilities(module._arch, native_kind, native_version):
            raise ValueError(f"AOT primitive requires {native_kind} version {native_version} on {module._arch}")
    if recipe.helpers:
        import numpy as np

        from taichi_forge._kernels import rle_mark_boundaries_and_starts_ndarray, rle_mark_boundaries_ndarray

        for entry, dtype, with_indices in recipe.helpers:
            # Zero-element host exemplars carry specialization metadata only.
            # Export neither allocates device arrays nor executes a helper.
            exemplar = {
                "keys": np.empty(0, dtype={"i32": "int32", "u32": "uint32", "i64": "int64", "u64": "uint64"}[dtype]),
                "flags": np.empty(0, dtype="int32"),
            }
            helper = rle_mark_boundaries_ndarray
            if with_indices:
                helper = rle_mark_boundaries_and_starts_ndarray
                exemplar["starts"] = exemplar["flags"]
            # Module.add_kernel rejects name conflicts and deduplicates the
            # identical specialization shared by multiple exported recipes.
            try:
                module.add_kernel(helper, template_args=exemplar, name=entry)
            except RuntimeError as error:
                if module._arch == _ti_core.vulkan and dtype in ("i64", "u64"):
                    raise RuntimeError(
                        "Cannot export the 64-bit unique helper. Vulkan requires "
                        "Module(caps=['spirv_has_int64']) and a device with shaderInt64. " + str(error)
                    ) from error
                raise
    module.add_execution_plan(name, recipe.plan)
    return recipe.plan
