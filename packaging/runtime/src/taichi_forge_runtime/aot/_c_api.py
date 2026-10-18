"""ctypes declarations for the public C API in c_api/include/taichi."""

from __future__ import annotations

import ctypes as C

from .. import _native

Handle = C.c_void_p
UInt = C.c_uint32

# Public TiCapability values, independent of the compiler's persisted enum.
CAPABILITY_NAMES = dict(
    enumerate(
        (
            "reserved",
            "spirv_version",
            "spirv_has_int8",
            "spirv_has_int16",
            "spirv_has_int64",
            "spirv_has_float16",
            "spirv_has_float64",
            "spirv_has_atomic_int64",
            "spirv_has_atomic_float16",
            "spirv_has_atomic_float16_add",
            "spirv_has_atomic_float16_minmax",
            "spirv_has_atomic_float",
            "spirv_has_atomic_float_add",
            "spirv_has_atomic_float_minmax",
            "spirv_has_atomic_float64",
            "spirv_has_atomic_float64_add",
            "spirv_has_atomic_float64_minmax",
            "spirv_has_variable_ptr",
            "spirv_has_physical_storage_buffer",
            "spirv_has_subgroup_basic",
            "spirv_has_subgroup_vote",
            "spirv_has_subgroup_arithmetic",
            "spirv_has_subgroup_ballot",
            "spirv_has_non_semantic_info",
            "spirv_has_no_integer_wrap_decoration",
            "spirv_has_int8_storage",
            "spirv_has_int16_storage",
            "spirv_has_storage_image_extended_formats",
            "spirv_has_ray_query",
            "spirv_has_cooperative_matrix",
            "cuda_compute_capability",
            "cuda_ptx_version",
            "spirv_has_sampled_image_array_non_uniform_indexing",
            "max_sampled_texture_collection_size",
            "vulkan_has_opacity_micromap",
            "vulkan_has_ray_tracing_pipeline",
        )
    )
)


class CapabilityLevelInfo(C.Structure):
    _fields_ = [("capability", UInt), ("level", UInt)]


class Shape(C.Structure):
    _fields_ = [("dim_count", UInt), ("dims", UInt * 16)]


class NdArray(C.Structure):
    _fields_ = [
        ("memory", Handle),
        ("shape", Shape),
        ("elem_shape", Shape),
        ("elem_type", UInt),
    ]


class ScalarValue(C.Union):
    _fields_ = [
        ("x8", C.c_uint8),
        ("x16", C.c_uint16),
        ("x32", UInt),
        ("x64", C.c_uint64),
    ]


class Scalar(C.Structure):
    _fields_ = [("type", UInt), ("value", ScalarValue)]


class TensorValue(C.Union):
    _fields_ = [
        ("x8", C.c_uint8 * 128),
        ("x16", C.c_uint16 * 64),
        ("x32", UInt * 32),
        ("x64", C.c_uint64 * 16),
    ]


class TensorContents(C.Structure):
    _fields_ = [("length", UInt), ("data", TensorValue)]


class Tensor(C.Structure):
    _fields_ = [("type", UInt), ("contents", TensorContents)]


class ImageExtent(C.Structure):
    _fields_ = [
        ("width", UInt),
        ("height", UInt),
        ("depth", UInt),
        ("array_layer_count", UInt),
    ]


class Texture(C.Structure):
    _fields_ = [
        ("image", Handle),
        ("sampler", Handle),
        ("dimension", UInt),
        ("extent", ImageExtent),
        ("format", UInt),
    ]


class ArgumentValue(C.Union):
    _fields_ = [
        ("native_resource", Handle),
        ("i32", C.c_int32),
        ("f32", C.c_float),
        ("ndarray", NdArray),
        ("texture", Texture),
        ("scalar", Scalar),
        ("tensor", Tensor),
    ]


class Argument(C.Structure):
    _fields_ = [("type", UInt), ("value", ArgumentValue)]


class NamedArgument(C.Structure):
    _fields_ = [("name", C.c_char_p), ("argument", Argument)]


class Gradient(C.Structure):
    _fields_ = [("argument_index", UInt), ("gradient", NdArray)]


class NamedGradient(C.Structure):
    _fields_ = [("name", C.c_char_p), ("gradient", NdArray)]


class Binding(C.Structure):
    _fields_ = [
        ("argument_index", UInt),
        ("layout_version", UInt),
        ("layout", Handle),
        ("gradient", C.POINTER(NdArray)),
    ]


class NamedBinding(C.Structure):
    _fields_ = [("name", C.c_char_p), ("layout_version", UInt), ("layout", Handle), ("gradient", C.POINTER(NdArray))]


class BindingFamilyInfo(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in ("identity", "bindings", "executables", "resource_bytes")]


class PreparedInfo(C.Structure):
    _fields_ = [
        ("dispatch_count", UInt),
        ("execution_kind", UInt),
        ("argument_bytes", C.c_uint64),
        ("launches", C.c_uint64),
    ]


class NativeOperationAvailability(C.Structure):
    _fields_ = [("capabilities", UInt), ("available", UInt), ("library_version", UInt)]


class PreparedMemoryInfo(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in ("resource_bytes", "native_resource_bytes", "workspace_bytes")]


class MemoryAllocateInfo(C.Structure):
    _fields_ = [
        ("size", C.c_uint64),
        ("host_write", UInt),
        ("host_read", UInt),
        ("export_sharing", UInt),
        ("usage", UInt),
    ]


class MemorySlice(C.Structure):
    _fields_ = [("memory", Handle), ("offset", C.c_uint64), ("size", C.c_uint64)]


class CudaExecutionDomainCreateInfo(C.Structure):
    _fields_ = [("capacity", UInt), ("instance_capacity", UInt), ("priority", C.c_int32)]


class CudaExecutionDomainInfo(C.Structure):
    _fields_ = [
        ("identity", C.c_uint64), ("owner", C.c_uint64),
        ("context", C.c_uint64), ("stream", C.c_uint64),
        ("requested_priority", C.c_int32), ("effective_priority", C.c_int32),
        ("least_priority", C.c_int32), ("greatest_priority", C.c_int32),
        ("capacity", C.c_uint64), ("pending", C.c_uint64),
        ("accepting", UInt), ("closed", UInt),
    ]


class CompletionInfo(C.Structure):
    _fields_ = [("owner", C.c_uint64), ("sequence", C.c_uint64), ("has_backend_work", UInt)]


class AotError(RuntimeError):
    """An error reported by the native AOT C API."""

    def __init__(self, operation, code, message):
        self.operation, self.code = operation, code
        super().__init__(f"{operation}: {message} (C API error {code})")


class Api:
    def __init__(self, library_path=None):
        self.library = _native.load_library(library_path)
        functions = {
            "ti_get_last_error": ([C.POINTER(C.c_uint64), C.c_void_p], C.c_int32),
            "ti_set_last_error": ([C.c_int32, C.c_char_p], None),
            "ti_create_runtime": ([UInt, UInt], Handle),
            "ti_get_runtime_capabilities": ([Handle, C.POINTER(UInt), C.POINTER(CapabilityLevelInfo)], None),
            "ti_configure_native_provider_ext": ([Handle, C.c_char_p, C.c_char_p, C.c_char_p], None),
            "ti_configure_native_library_ext": ([Handle, C.c_char_p, C.c_char_p], None),
            "ti_get_native_operation_capabilities_ext": ([Handle, C.c_char_p, UInt], UInt),
            "ti_get_native_resource_capabilities_ext": ([Handle, C.c_char_p, UInt], UInt),
            "ti_probe_native_operation_ext": ([Handle, C.c_char_p, UInt, C.POINTER(NativeOperationAvailability)], None),
            "ti_probe_native_resource_ext": ([Handle, C.c_char_p, UInt, C.POINTER(NativeOperationAvailability)], None),
            "ti_get_prepared_execution_resource_bytes_ext": ([Handle, Handle], C.c_uint64),
            "ti_get_prepared_execution_memory_info_ext": ([Handle, Handle, C.POINTER(PreparedMemoryInfo)], None),
            "ti_destroy_runtime": ([Handle], None),
            "ti_wait": ([Handle], None),
            "ti_allocate_memory": ([Handle, C.POINTER(MemoryAllocateInfo)], Handle),
            "ti_free_memory": ([Handle, Handle], None),
            "ti_copy_memory_from_host_ext": (
                [Handle, C.POINTER(MemorySlice), C.c_void_p],
                None,
            ),
            "ti_copy_memory_to_host_ext": (
                [Handle, C.c_void_p, C.POINTER(MemorySlice)],
                None,
            ),
            "ti_load_aot_module": ([Handle, C.c_char_p], Handle),
            "ti_destroy_aot_module": ([Handle], None),
            "ti_get_aot_module_kernel": ([Handle, C.c_char_p], Handle),
            "ti_get_aot_module_compute_graph": ([Handle, C.c_char_p], Handle),
            "ti_prepare_kernel_ext": ([Handle, Handle, UInt, C.POINTER(Argument), UInt, C.POINTER(Binding)], Handle),
            "ti_prepare_compute_graph_ext": (
                [Handle, Handle, UInt, C.POINTER(NamedArgument), UInt, C.POINTER(NamedBinding)],
                Handle,
            ),
            "ti_launch_prepared_ext": ([Handle, Handle], None),
            "ti_initialize_prepared_ext": ([Handle, Handle], None),
            "ti_prepare_execution_plan_ext": (
                [Handle, Handle, C.c_uint64, C.c_char_p, UInt, C.POINTER(NamedArgument), UInt, C.POINTER(NamedBinding)],
                Handle,
            ),
            "ti_destroy_prepared_ext": ([Handle, Handle], None),
            "ti_create_native_resource_ext": (
                [Handle, C.c_uint64, C.c_char_p, UInt, C.POINTER(NamedArgument), C.POINTER(UInt)], Handle,
            ),
            "ti_destroy_native_resource_ext": ([Handle, Handle], None),
            "ti_create_host_blob_ext": ([Handle, C.c_uint64, C.c_void_p], Handle),
            "ti_prepare_execution_binding_ext": (
                [Handle, Handle, UInt, C.POINTER(NamedArgument), UInt, C.POINTER(NamedBinding)],
                Handle,
            ),
            "ti_get_prepared_binding_family_info_ext": ([Handle, Handle, C.POINTER(BindingFamilyInfo)], None),
            "ti_get_prepared_execution_info_ext": ([Handle, Handle, C.POINTER(PreparedInfo)], None),
            "ti_get_prepared_execution_workspace_bytes_ext": ([Handle, Handle], C.c_uint64),
            "ti_launch_kernel_with_gradients_ext": (
                [Handle, Handle, UInt, C.POINTER(Argument), UInt, C.POINTER(Gradient)],
                None,
            ),
            "ti_launch_compute_graph_with_gradients_ext": (
                [
                    Handle,
                    Handle,
                    UInt,
                    C.POINTER(NamedArgument),
                    UInt,
                    C.POINTER(NamedGradient),
                ],
                None,
            ),
        }
        for name, (arguments, result) in functions.items():
            try:
                function = getattr(self.library, name)
            except AttributeError as error:
                raise ImportError(
                    "The selected Forge runtime lacks the AOT C API; build with TI_WITH_RUNTIME_C_API=ON"
                ) from error
            function.argtypes, function.restype = arguments, result
            setattr(self, name, function)

    def load_cuda_domains(self):
        if getattr(self, "_cuda_domains_loaded", False):
            return
        declarations = {
            "ti_create_cuda_execution_domain_ext": ([Handle, C.POINTER(CudaExecutionDomainCreateInfo)], Handle),
            "ti_get_cuda_execution_domain_info_ext": ([Handle, C.POINTER(CudaExecutionDomainInfo)], None),
            "ti_stop_cuda_execution_domain_ext": ([Handle], None),
            "ti_close_cuda_execution_domain_ext": ([Handle], None),
            "ti_release_cuda_execution_domain_ext": ([Handle], None),
            "ti_collect_cuda_execution_domain_ext": ([Handle, UInt], UInt),
            "ti_get_cuda_execution_domain_completion_ext": ([Handle], Handle),
            "ti_query_completion_ext": ([Handle], UInt),
            "ti_wait_completion_ext": ([Handle], None),
            "ti_get_completion_info_ext": ([Handle, C.POINTER(CompletionInfo)], None),
            "ti_release_completion_ext": ([Handle], None),
        }
        try:
            functions = {name: getattr(self.library, name) for name in declarations}
        except AttributeError as error:
            raise NotImplementedError("The selected Forge runtime has no CUDA execution-domain API") from error
        for name, function in functions.items():
            function.argtypes, function.restype = declarations[name]
            setattr(self, name, function)
        self._cuda_domains_loaded = True

    def call(self, function, *arguments):
        self.ti_set_last_error(0, None)
        result = function(*arguments)
        length = C.c_uint64()
        code = self.ti_get_last_error(C.byref(length), None)
        if code:
            message = C.create_string_buffer(length.value or 1)
            self.ti_get_last_error(C.byref(length), message)
            raise AotError(function.__name__, code, message.value.decode("utf-8", errors="replace"))
        return result
