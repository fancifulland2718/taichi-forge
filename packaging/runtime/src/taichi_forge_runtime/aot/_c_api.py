"""ctypes declarations for the public C API in c_api/include/taichi."""

from __future__ import annotations

import ctypes as C

from .. import _native

Handle = C.c_void_p
UInt = C.c_uint32


class CompletionSource(C.Structure):
    _fields_ = [("kind", UInt), ("handle", Handle)]


class CompletionWaitResult(C.Structure):
    _fields_ = [("status", UInt), ("index", C.c_uint64)]

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
            "spirv_has_shared_atomic_int64",
            "spirv_has_shared_atomic_float16_add",
            "spirv_has_shared_atomic_float16_minmax",
            "spirv_has_shared_atomic_float_add",
            "spirv_has_shared_atomic_float_minmax",
            "spirv_has_shared_atomic_float64_add",
            "spirv_has_shared_atomic_float64_minmax",
        )
    )
)


class CapabilityLevelInfo(C.Structure):
    _fields_ = [("capability", UInt), ("level", UInt)]


class CudaExternalMemoryImportInfo(C.Structure):
    _fields_ = [
        ("handle_type", UInt),
        ("dedicated", UInt),
        ("memory_handle", C.c_uint64),
        ("ready_for_cuda", C.c_uint64),
        ("ready_for_external", C.c_uint64),
        ("allocation_size", C.c_uint64),
        ("byte_offset", C.c_uint64),
        ("byte_size", C.c_uint64),
        ("device_uuid", C.c_uint8 * 16),
    ]


class VulkanExternalBufferImportInfo(C.Structure):
    _fields_ = [
        ("struct_size", UInt), ("schema_version", UInt),
        ("memory_handle", C.c_uint64),
        ("ready_for_consumer", C.c_uint64),
        ("ready_for_external", C.c_uint64),
        ("buffer_size", C.c_uint64), ("allocation_size", C.c_uint64),
        ("bind_offset", C.c_uint64),
        ("memory_type_index", UInt), ("buffer_flags", UInt),
        ("buffer_usage", UInt), ("allocation_flags", UInt),
        ("sharing_mode", UInt), ("dedicated", UInt),
        ("device_uuid", C.c_uint8 * 16), ("driver_uuid", C.c_uint8 * 16),
        ("byte_offset", C.c_uint64), ("byte_size", C.c_uint64),
    ]


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


class ValueArgument(C.Structure):
    _fields_ = [("size", C.c_uint64), ("data", C.c_void_p)]


class ArgumentValue(C.Union):
    _fields_ = [
        ("native_resource", Handle),
        ("i32", C.c_int32),
        ("f32", C.c_float),
        ("ndarray", NdArray),
        ("texture", Texture),
        ("scalar", Scalar),
        ("tensor", Tensor),
        ("value", ValueArgument),
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


class NdArrayLayoutInfo(C.Structure):
    _fields_ = [("version", UInt), ("member_count", UInt), ("element_stride", C.c_uint64), ("alignment", C.c_uint64)]


class NdArrayLayoutMember(C.Structure):
    _fields_ = [
        ("path", C.c_char_p), ("scalar_type", UInt), ("element_shape", Shape),
        ("offset", C.c_uint64), ("size", C.c_uint64), ("alignment", C.c_uint64),
    ]


class KernelReturnInfo(C.Structure):
    _fields_ = [("buffer_size", C.c_uint64), ("layout_json", C.c_char_p)]


class KernelArgumentInfo(C.Structure):
    _fields_ = [("argument_count", UInt), ("layout_json", C.c_char_p)]


class KernelArgPackInfo(C.Structure):
    _fields_ = [("layout_json", C.c_char_p)]


class ArgPackMember(C.Structure):
    _fields_ = [
        ("path_count", UInt), ("path", C.POINTER(UInt)), ("argument", Argument),
        ("layout_version", UInt), ("layout", Handle), ("gradient", C.POINTER(NdArray)),
    ]


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


class VulkanExecutionDomainCreateInfo(C.Structure):
    _fields_ = [("capacity", UInt), ("instance_capacity", UInt), ("gpu_timing", UInt)]


class VulkanExecutionDomainInfo(C.Structure):
    _fields_ = [
        ("identity", C.c_uint64), ("owner", C.c_uint64), ("queue_index", UInt),
        ("capacity", C.c_uint64), ("pending", C.c_uint64),
        ("accepting", UInt), ("closed", UInt),
    ]


class CompletionInfo(C.Structure):
    _fields_ = [("owner", C.c_uint64), ("sequence", C.c_uint64), ("has_backend_work", UInt)]


class CompletionGpuTiming(C.Structure):
    _fields_ = [("status", UInt), ("scope", UInt), ("owner", C.c_uint64),
                ("sequence", C.c_uint64), ("duration_ns", C.c_uint64)]


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
            "ti_get_kernel_ndarray_layout_ext": ([Handle, Handle, UInt], Handle),
            "ti_get_aot_module_ndarray_layout_ext": ([Handle, Handle, C.c_char_p], Handle),
            "ti_get_compute_graph_ndarray_layout_ext": ([Handle, Handle, C.c_char_p], Handle),
            "ti_get_ndarray_layout_info_ext": ([Handle, Handle, C.POINTER(NdArrayLayoutInfo)], None),
            "ti_get_ndarray_layout_member_ext": ([Handle, Handle, UInt, C.POINTER(NdArrayLayoutMember)], None),
            "ti_get_kernel_return_info_ext": ([Handle, Handle, C.POINTER(KernelReturnInfo)], None),
            "ti_get_kernel_argument_info_ext": ([Handle, Handle, C.POINTER(KernelArgumentInfo)], None),
            "ti_get_kernel_argpack_info_ext": ([Handle, Handle, UInt, C.POINTER(KernelArgPackInfo)], None),
            "ti_get_compute_graph_argpack_kernel_ext": ([Handle, Handle, C.c_char_p, C.POINTER(UInt)], Handle),
            "ti_create_kernel_argpack_ext": ([Handle, Handle, UInt, UInt, C.POINTER(ArgPackMember)], Handle),
            "ti_launch_kernel_with_result_ext": (
                [Handle, Handle, UInt, C.POINTER(Argument), UInt, C.POINTER(Binding), C.c_uint64, C.c_void_p], None,
            ),
            "ti_launch_kernel_with_bindings_ext": (
                [Handle, Handle, UInt, C.POINTER(Argument), UInt, C.POINTER(Binding)], None,
            ),
            "ti_launch_compute_graph_with_bindings_ext": (
                [Handle, Handle, UInt, C.POINTER(NamedArgument), UInt, C.POINTER(NamedBinding)], None,
            ),
            "ti_prepare_kernel_ext": ([Handle, Handle, UInt, C.POINTER(Argument), UInt, C.POINTER(Binding)], Handle),
            "ti_prepare_compute_graph_ext": (
                [Handle, Handle, UInt, C.POINTER(NamedArgument), UInt, C.POINTER(NamedBinding)],
                Handle,
            ),
            "ti_launch_prepared_ext": ([Handle, Handle], None),
            "ti_read_prepared_kernel_result_ext": ([Handle, Handle, C.c_uint64, C.c_void_p], None),
            "ti_submit_prepared_ext": ([Handle, Handle], Handle),
            "ti_query_submission_ext": ([Handle], UInt),
            "ti_wait_submission_ext": ([Handle], None),
            "ti_destroy_submission_ext": ([Handle], None),
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

    def load_cuda_external_memory(self):
        if getattr(self, "ti_import_cuda_external_memory_ext", None) is not None:
            return
        try:
            function = self.library.ti_import_cuda_external_memory_ext
        except AttributeError as error:
            raise NotImplementedError("The selected Forge runtime has no synchronized CUDA external-memory API") from error
        function.argtypes, function.restype = [Handle, C.POINTER(CudaExternalMemoryImportInfo)], Handle
        self.ti_import_cuda_external_memory_ext = function

    def load_cuda_external_ready_gate(self):
        from ._external_ready import _ReadyGate

        name = "ti_import_cuda_external_memory_with_ready_gate_ext"
        if getattr(self, name, None) is not None:
            return
        try:
            function = getattr(self.library, name)
        except AttributeError as error:
            raise NotImplementedError("The selected Forge runtime lacks native external ready gates") from error
        function.argtypes = [Handle, C.POINTER(CudaExternalMemoryImportInfo), C.POINTER(_ReadyGate)]
        function.restype = Handle
        setattr(self, name, function)

    def load_vulkan_external_buffer(self):
        from ._external_ready import _ReadyGate

        name = "ti_import_vulkan_external_buffer_ext"
        if getattr(self, name, None) is not None:
            return
        try:
            function = getattr(self.library, name)
        except AttributeError as error:
            raise NotImplementedError("The selected Forge runtime lacks managed Vulkan external buffers") from error
        function.argtypes = [Handle, C.POINTER(VulkanExternalBufferImportInfo), C.POINTER(_ReadyGate)]
        function.restype = Handle
        setattr(self, name, function)

    def load_ready_submit(self):
        name = "ti_try_submit_prepared_ready_ext"
        if getattr(self, name, None) is not None:
            return
        try:
            function = getattr(self.library, name)
        except AttributeError as error:
            raise NotImplementedError("The selected Forge runtime lacks ready-only domain submission") from error
        function.argtypes = [Handle, Handle, UInt, C.POINTER(Handle), C.POINTER(Handle)]
        function.restype = UInt
        setattr(self, name, function)

    def load_completion_sets(self):
        if getattr(self, "_completion_sets_loaded", False):
            return
        declarations = {
            "ti_create_completion_set_ext": ([C.c_uint64, C.POINTER(CompletionSource), C.c_int64], Handle),
            "ti_wait_completion_set_ext": ([Handle, C.c_int64, C.POINTER(CompletionWaitResult)], None),
            "ti_cancel_completion_set_ext": ([Handle], None),
            "ti_release_completion_set_ext": ([Handle], None),
        }
        try:
            functions = {name: getattr(self.library, name) for name in declarations}
        except AttributeError as error:
            raise NotImplementedError("The selected runtime has no completion wait-any API") from error
        for name, function in functions.items():
            function.argtypes, function.restype = declarations[name]
            setattr(self, name, function)
        self._completion_sets_loaded = True

    def load_cuda_domains(self):
        self.load_execution_domains("cuda")

    def load_vulkan_domains(self):
        self.load_execution_domains("vulkan")

    def load_execution_domains(self, backend):
        if backend not in ("cuda", "vulkan"):
            raise ValueError("Unsupported execution domain backend")
        marker = "_" + backend + "_domains_loaded"
        if getattr(self, marker, False):
            return
        create_info = CudaExecutionDomainCreateInfo if backend == "cuda" else VulkanExecutionDomainCreateInfo
        info = CudaExecutionDomainInfo if backend == "cuda" else VulkanExecutionDomainInfo
        plan_arguments = [Handle, Handle, Handle, C.c_uint64, C.c_char_p, UInt,
                          C.POINTER(NamedArgument), UInt, C.POINTER(NamedBinding)]
        declarations = {
            f"ti_create_{backend}_execution_domain_ext": ([Handle, C.POINTER(create_info)], Handle),
            f"ti_get_{backend}_execution_domain_info_ext": ([Handle, C.POINTER(info)], None),
            f"ti_stop_{backend}_execution_domain_ext": ([Handle], None),
            f"ti_close_{backend}_execution_domain_ext": ([Handle], None),
            f"ti_release_{backend}_execution_domain_ext": ([Handle], None),
            f"ti_get_prepared_{backend}_instance_ext": ([Handle, Handle], Handle),
            f"ti_retire_{backend}_execution_instance_ext": ([Handle], None),
            f"ti_release_{backend}_execution_instance_ext": ([Handle], None),
            "ti_finish_prepared_instance_retirement_ext": ([Handle, Handle], None),
            f"ti_collect_{backend}_execution_domain_ext": ([Handle, UInt], UInt),
            f"ti_get_{backend}_execution_domain_completion_ext": ([Handle], Handle),
            "ti_try_submit_prepared_ext": ([Handle, Handle, UInt, C.POINTER(Handle), C.POINTER(Handle)], UInt),
            "ti_query_completion_ext": ([Handle], UInt),
            "ti_wait_completion_ext": ([Handle], None),
            "ti_get_completion_info_ext": ([Handle, C.POINTER(CompletionInfo)], None),
            "ti_release_completion_ext": ([Handle], None),
        }
        if backend == "cuda":
            declarations.update({
                "ti_create_timed_cuda_execution_domain_ext": ([Handle, C.POINTER(create_info)], Handle),
                "ti_prepare_execution_plan_on_domain_ext": (plan_arguments, Handle),
                "ti_prepare_execution_plan_parameters_on_domain_ext": (
                    plan_arguments + [UInt, C.POINTER(C.c_char_p)], Handle),
            })
        else:
            declarations["ti_prepare_execution_plan_on_vulkan_domain_ext"] = (
                plan_arguments + [UInt, C.POINTER(C.c_char_p)], Handle)
        try:
            functions = {name: getattr(self.library, name) for name in declarations}
        except AttributeError as error:
            raise NotImplementedError(f"The selected Forge runtime has no {backend} execution-domain API") from error
        for name, function in functions.items():
            function.argtypes, function.restype = declarations[name]
            setattr(self, name, function)
        setattr(self, marker, True)

    def load_submission_parameters(self):
        if getattr(self, "_submission_parameters_loaded", False):
            return
        declarations = {
            "ti_try_submit_prepared_parameters_ext": (
                [Handle, Handle, UInt, C.POINTER(Handle), C.POINTER(Handle), UInt, UInt,
                 C.POINTER(C.c_uint64)], UInt,
            ),
        }
        try:
            functions = {name: getattr(self.library, name) for name in declarations}
        except AttributeError as error:
            raise NotImplementedError("The selected runtime has no submission parameter API") from error
        for name, function in functions.items():
            function.argtypes, function.restype = declarations[name]
            setattr(self, name, function)
        self._submission_parameters_loaded = True

    def load_completion_timing(self):
        if getattr(self, "_completion_timing_loaded", False):
            return
        declarations = {
            "ti_get_completion_gpu_timing_ext": ([Handle, C.POINTER(CompletionGpuTiming)], None),
            "ti_try_submit_prepared_sampled_ext": (
                [Handle, Handle, UInt, C.POINTER(Handle), C.POINTER(Handle), UInt, UInt,
                 UInt, C.POINTER(C.c_uint64)], UInt),
        }
        try:
            functions = {name: getattr(self.library, name) for name in declarations}
        except AttributeError as error:
            raise NotImplementedError("The selected runtime has no completion GPU timing API") from error
        for name, function in functions.items():
            function.argtypes, function.restype = declarations[name]
            setattr(self, name, function)
        self._completion_timing_loaded = True

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
