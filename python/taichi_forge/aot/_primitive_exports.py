"""Producer recipes expand once into ordinary kernels and native operations.

No backend implementations live here. The common native registry determines
compiled support; selected helpers use the module's ordinary AOT compiler.
Only the expanded plan/code are deployed, never this registry or Python calls.
"""

from dataclasses import dataclass

from taichi_forge._lib import core as _ti_core
from taichi_forge.algorithms._consecutive_definition import unique_helper


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
        for entry, dtype, with_indices in recipe.helpers:
            # Zero-element host exemplars carry specialization metadata only.
            # Export neither allocates device arrays nor executes a helper.
            helper, exemplar = unique_helper(dtype, with_indices)
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


def export_host_unique_recipes(module, sources, projections):
    """Compile the existing run-boundary helper for retained Host calls and their key types."""
    import ast
    from ._package_types import qualified_kind

    supported = {"i32", "u32", "i64", "u64"}
    required = set()
    for name, _, _, _, _, kind, schemas, _ in sources:
        # Host projection excludes numerical bodies and producer-only paths. Its release
        # after module assembly still follows the normal streaming compilation lifetime.
        host, _, _, requirements = projections[name]
        if not host.uses_algorithms:
            continue
        projected = host.tree
        methods = {"CompactWorkspace", "RunLengthWorkspace", "sort_by_key", "compact", "unique_by_key",
                   "prepare_sort", "prepare_compact", "prepare_compact_columns", "prepare_unique_by_key"}
        if any(isinstance(node, ast.Attribute) and node.attr in methods for node in ast.walk(projected)):
            requirements.append("host_compact_unique_v1")
        # If the input is a plain typed owner field, retain just that helper.
        # A Host parameter can carry any supported integer key type at binding.
        owners = {}
        for cls in projected.body:
            if isinstance(cls, ast.ClassDef):
                schema = schemas.get(cls.name, {})
                owners.update((id(call), schema) for call in ast.walk(cls) if isinstance(call, ast.Call))
        for call in ast.walk(projected):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                continue
            if call.func.attr not in ("unique_by_key", "prepare_unique_by_key"):
                continue
            keys = call.args[0] if call.args else next((kw.value for kw in call.keywords if kw.arg == "keys"), None)
            declaration = (owners.get(id(call), {}).get(keys.attr, {}) if isinstance(keys, ast.Attribute)
                           and isinstance(keys.value, ast.Name) and keys.value.id == "self" else {})
            dtype = declaration.get("type_node")
            fixed = qualified_kind(dtype, kind) if dtype is not None and not declaration.get("dtype_policy") else None
            required.update((fixed,) if fixed in supported else supported)
    for dtype in sorted(required):
        module.add_primitive("__forge_package_unique_by_key_" + dtype, "unique", key_dtype=dtype,
                             values="values", unique_values="unique_values")
