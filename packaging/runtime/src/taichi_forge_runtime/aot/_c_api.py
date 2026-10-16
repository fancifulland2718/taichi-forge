"""ctypes declarations for the public C API in c_api/include/taichi."""

from __future__ import annotations

import ctypes as C

from .. import _native

Handle = C.c_void_p
UInt = C.c_uint32


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


class PreparedInfo(C.Structure):
    _fields_ = [
        ("dispatch_count", UInt),
        ("execution_kind", UInt),
        ("argument_bytes", C.c_uint64),
        ("launches", C.c_uint64),
    ]


class NativeOperationAvailability(C.Structure):
    _fields_ = [("capabilities", UInt), ("available", UInt), ("library_version", UInt)]


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
            "ti_configure_native_provider_ext": ([Handle, C.c_char_p, C.c_char_p, C.c_char_p], None),
            "ti_configure_native_library_ext": ([Handle, C.c_char_p, C.c_char_p], None),
            "ti_get_native_operation_capabilities_ext": ([Handle, C.c_char_p, UInt], UInt),
            "ti_probe_native_operation_ext": ([Handle, C.c_char_p, UInt, C.POINTER(NativeOperationAvailability)], None),
            "ti_get_prepared_execution_resource_bytes_ext": ([Handle, Handle], C.c_uint64),
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
