"""Optional program ABI; kept separate from the legacy fixed-query table."""

import ctypes as c


class Module(c.Structure):
    _fields_ = [("ptx", c.c_char_p), ("ptx_size", c.c_size_t)]


class Entry(c.Structure):
    _fields_ = [("module_index", c.c_uint32), ("name", c.c_char_p)]


class Group(c.Structure):
    _fields_ = [("kind", c.c_uint32), ("entry", Entry), ("any_hit", Entry)]


class ProgramDesc(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("module_count", c.c_uint32),
        ("modules", c.POINTER(Module)),
        ("group_count", c.c_uint32),
        ("groups", c.POINTER(Group)),
        ("parameter_name", c.c_char_p),
        ("parameter_size", c.c_uint32),
        ("payload_count", c.c_uint32),
        ("attribute_count", c.c_uint32),
        ("max_trace_depth", c.c_uint32),
        ("allow_opacity_micromaps", c.c_uint32),
    ]


class Record(c.Structure):
    _fields_ = [("group_index", c.c_uint32), ("data_size", c.c_uint32), ("data", c.c_void_p)]


class Scene(c.Structure):
    _fields_ = [("scene", c.c_void_p), ("kind", c.c_uint32)]


class SceneInfo(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("max_sbt_offset", c.c_uint32),
        ("traversable", c.c_uint64),
        ("has_opacity_micromaps", c.c_uint32),
    ]


class LaunchDesc(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("width", c.c_uint32),
        ("height", c.c_uint32),
        ("depth", c.c_uint32),
        ("parameters", c.c_void_p),
        ("parameter_size", c.c_uint32),
        ("raygen", Record),
        ("miss", c.POINTER(Record)),
        ("miss_count", c.c_uint32),
        ("hit", c.POINTER(Record)),
        ("hit_count", c.c_uint32),
        ("scenes", c.POINTER(Scene)),
        ("scene_count", c.c_uint32),
    ]


class Memory(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("continuation_stack_bytes", c.c_uint32),
        ("device_data_bytes", c.c_uint64),
        ("pinned_host_bytes", c.c_uint64),
        ("parameter_bytes", c.c_uint32),
        ("miss_stride", c.c_uint32),
        ("hit_stride", c.c_uint32),
    ]


class ProgramApi(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("revision", c.c_uint32),
        ("create", c.CFUNCTYPE(c.c_int, c.c_void_p, c.POINTER(ProgramDesc), c.POINTER(c.c_void_p))),
        ("destroy", c.CFUNCTYPE(c.c_int, c.c_void_p)),
        ("prepare", c.CFUNCTYPE(c.c_int, c.c_void_p, c.POINTER(LaunchDesc), c.POINTER(c.c_void_p))),
        ("initialize", c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_uint64)),
        ("launch", c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_uint64)),
        ("destroy_launch", c.CFUNCTYPE(c.c_int, c.c_void_p)),
        ("memory", c.CFUNCTYPE(c.c_int, c.c_void_p, c.POINTER(Memory))),
        ("scene_info", c.CFUNCTYPE(c.c_int, c.c_void_p, c.POINTER(Scene), c.POINTER(SceneInfo))),
    ]


def load_program_api(provider):
    from taichi_forge.hardware._optix import _api_has, _invoke_checked
    from taichi_forge.lang.exception import TaichiRuntimeError

    api = provider._loaded.api
    if not (int(api.info.features) & (1 << 14)) or not _api_has(api, "get_program_api"):
        raise TaichiRuntimeError("this OptiX adapter lacks programmable pipelines; use a newer Forge adapter")
    result = ProgramApi()
    _invoke_checked(api, api.get_program_api, c.sizeof(result), c.byref(result))
    if (
        result.struct_size < c.sizeof(result)
        or result.revision != 1
        or any(not getattr(result, name) for name, _ in ProgramApi._fields_[2:])
    ):
        raise TaichiRuntimeError("OptiX programmable pipeline API is truncated or incompatible")
    return result
