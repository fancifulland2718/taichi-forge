"""Ray acceleration capability catalog fragments."""

def ray_command_operations(_operation):
    return (
        _operation(
            "ray.as_build.vulkan",
            "ray.acceleration_structure",
            "vulkan_ray",
            ("vulkan",),
            "core",
            "fixed_function",
            "native_command",
            "implementation_defined",
            ("python", "graph"),
            "native_command",
            "unsupported",
            "runtime_ordered",
            "provider_owned",
            "existing_public",
            activation_mode="explicit_hardware_api",
            resource_effects=(
                "read:geometry",
                "write:acceleration_structure",
                "write:scratch",
            ),
            lifetime_policy="resource_generation",
            update_policy="rebuild",
            requirements=("VK_KHR_acceleration_structure",),
            public_api=(
                "ti.hardware.ray.TriangleBLAS / "
                "ti.hardware.ray.InstanceTLAS"
            ),
            dtypes=("vertex:f32", "index:i32"),
            layouts=("scalar (N,3)", "AOS vector-3 (N,)"),
            notes=(
                "Independent fixed-topology triangle BLAS and fixed-order instance TLAS resources.",
                "Instance descriptors expose row-major 3x4 transforms, 8-bit masks, and 24-bit custom indices.",
                "TriangleScene remains the identity-instance compatibility wrapper.",
            ),
        ),
        _operation(
            "ray.as_refit.vulkan",
            "ray.acceleration_structure",
            "vulkan_ray",
            ("vulkan",),
            "core",
            "fixed_function",
            "native_command",
            "implementation_defined",
            ("python", "graph"),
            "native_command",
            "recordable",
            "runtime_ordered",
            "provider_owned",
            "existing_public",
            activation_mode="explicit_hardware_api",
            resource_effects=(
                "read:geometry",
                "read_write:acceleration_structure",
                "write:scratch",
            ),
            lifetime_policy="resource_generation",
            update_policy="refit",
            requirements=("VK_KHR_acceleration_structure",),
            public_api=(
                "ti.hardware.ray.TriangleBLAS.refit / "
                "ti.hardware.ray.InstanceTLAS.refit"
            ),
            dtypes=("vertex:f32",),
            layouts=("scalar (N,3)", "AOS vector-3 (N,)"),
            notes=(
                "Explicit Python or Graph native command; never selected by an ordinary kernel.",
                "BLAS refit is vertex-only; TLAS refit may update transforms, masks, and custom indices.",
                "BLAS counts and TLAS BLAS count/order remain fixed for the resource lifetime.",
                "TriangleScene retains its identity-TLAS compatibility refit route.",
            ),
        ),
        _operation(
            "ray.query.batch.vulkan",
            "ray.query",
            "vulkan_ray",
            ("vulkan",),
            "core",
            "fixed_function",
            "native_shader_operation",
            "implementation_defined",
            ("python", "graph"),
            "native_command",
            "recordable",
            "runtime_ordered",
            "provider_owned",
            "existing_public",
            activation_mode="explicit_hardware_api",
            resource_effects=(
                "read:acceleration_structure",
                "read:rays",
                "write:hits",
            ),
            lifetime_policy="resource_generation",
            update_policy="rebind",
            requirements=("VK_KHR_acceleration_structure", "VK_KHR_ray_query"),
            public_api=(
                "ti.hardware.ray.InstanceTLAS.trace / "
                "ti.hardware.ray.TriangleScene.trace"
            ),
            dtypes=("ray:f32", "hit:f32"),
            shapes_or_tiles=("rays:(N,8)", "hits:(N,4)", "workgroup:128"),
            layouts=("scalar 2D", "AOS vector"),
            numeric_contracts=(
                "ray:[origin.xyz,tmin,direction.xyz,tmax]",
                "hit:[t,primitive_id,instance_id,hit_flag]",
                "miss:[-1,-1,-1,0]",
            ),
            notes=(
                "Explicit Python or Graph native command; never selected by an ordinary kernel.",
            ),
        ),
        _operation(
            "ray.query.inline.vulkan",
            "ray.query",
            "vulkan_ray",
            ("vulkan",),
            "core",
            "fixed_function",
            "native_shader_operation",
            "implementation_defined",
            ("kernel",),
            "kernel_intrinsic",
            "inline",
            "runtime_ordered",
            "none",
            "planned",
            activation_mode="explicit_kernel_intrinsic",
            resource_effects=("read:acceleration_structure",),
            lifetime_policy="runtime_generation",
            update_policy="immutable",
            requirements=(
                "acceleration-structure kernel argument and lifetime/effect contract",
                "typed RayQuery IR and structured query control",
                "SPV_KHR_ray_query type/instruction lowering and descriptor binding",
                "VK_KHR_ray_query device feature",
            ),
            public_api="ti.hardware.ray",
            notes=(
                "The existing TriangleScene batch provider uses a separate embedded SPIR-V shader; it does not supply a kernel-visible AS value or RayQuery IR.",
                "Batch direct/root-Graph query remains the qualified route until the complete inline compiler and resource-binding chain exists.",
            ),
        ),
    )


def ray_optional_operations(_operation):
    return (
        _operation(
            "ray.query.batch.optix",
            "ray.query",
            "optix",
            ("cuda",),
            "lazy_external",
            "vendor_hardware_runtime",
            "vendor_hardware_runtime",
            "implementation_defined",
            ("python", "graph"),
            "external_library",
            "opaque",
            "explicit",
            "provider_owned",
            "planned",
            activation_mode="explicit_hardware_api",
            dependency_name="OptiX",
            resource_effects=("read:scene", "read:rays", "write:hits"),
            lifetime_policy="provider_plan",
            update_policy="rebuild",
            requirements=(
                "user-provided licensed OptiX SDK headers for a qualified OPTIX_ABI_VERSION",
                "lazy optixQueryFunctionTable loader with ABI isolation",
                "OptiX module/program-group/pipeline/SBT and GAS/IAS resource contracts",
                "qualified device-program build or artifact strategy",
            ),
            public_api="ti.hardware.ray",
            notes=(
                "OptiX function-table layout and initialization are SDK-header/ABI defined; a shared library name alone is not a safe provider contract.",
                "Keep this as a user-built plugin/source-build candidate until header licensing, ABI coverage, device programs, and pipeline lifetime are closed.",
            ),
        ),
    )


__all__ = (
    "ray_command_operations",
    "ray_optional_operations",
)
