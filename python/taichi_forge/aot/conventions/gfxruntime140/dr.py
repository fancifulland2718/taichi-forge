"""
Data representation of all JSON data structures following the GfxRuntime140
convention.
"""

from typing import Any, Dict, List, Optional

from taichi_forge.aot.utils import dump_json_data_model, json_data_model


@json_data_model
class FieldAttributes:
    def __init__(self, j: Dict[str, Any]) -> None:
        dtype = j["dtype"]
        dtype_name = j["dtype_name"]
        element_shape = j["element_shape"]
        field_name = j["field_name"]
        is_scalar = j["is_scalar"]
        mem_offset_in_parent = j["mem_offset_in_parent"]
        shape = j["shape"]

        self.dtype: int = int(dtype)
        self.dtype_name: str = str(dtype_name)
        self.element_shape: List[int] = [int(x) for x in element_shape]
        self.field_name: str = str(field_name)
        self.is_scalar: bool = bool(is_scalar)
        self.mem_offset_in_parent: int = int(mem_offset_in_parent)
        self.shape: List[int] = [int(x) for x in shape]


@json_data_model
class ArgumentAttributes:
    def __init__(self, j: Dict[str, Any]) -> None:
        index = j["key"][0]
        dtype = j["value"]["dtype"]
        element_shape = j["value"]["element_shape"]
        field_dim = j["value"]["field_dim"]
        fmt = j["value"]["format"]
        is_array = j["value"]["is_array"]
        offset_in_mem = j["value"]["offset_in_mem"]
        stride = j["value"]["stride"]
        # (penguinliong) Note that the name field is optional for kernels.
        # Kernels are always launched by indexed arguments and this is for
        # debugging and header generation only.
        name = j["value"]["name"] if "name" in j["value"] and len(j["value"]["name"]) > 0 else None
        ptype = j["value"]["ptype"] if "ptype" in j["value"] else None

        self.dtype: int = int(dtype)
        self.element_shape: List[int] = [int(x) for x in element_shape]
        self.field_dim: int = int(field_dim)
        self.format: int = int(fmt)
        self.index: int = int(index)
        self.is_array: bool = bool(is_array)
        self.offset_in_mem: int = int(offset_in_mem)
        self.stride: int = int(stride)
        self.name: Optional[str] = str(name) if name is not None else None
        self.ptype: Optional[int] = int(ptype) if ptype is not None else None


@json_data_model
class ContextAttributes:
    def __init__(self, j: Dict[str, Any]) -> None:
        arg_attribs_vec_ = j["arg_attribs_vec_"]
        args_bytes_ = j["args_bytes_"]
        arr_access = j["arr_access"]
        ret_attribs_vec_ = j["ret_attribs_vec_"]
        rets_bytes_ = j["rets_bytes_"]

        self.arg_attribs_vec_: List[ArgumentAttributes] = [ArgumentAttributes(x) for x in arg_attribs_vec_]
        self.arg_attribs_vec_.sort(key=lambda x: x.index)
        self.args_bytes_: int = int(args_bytes_)
        self.arr_access: List[int] = [int(x["value"]) for x in arr_access]
        self.ret_attribs_vec_: List[ArgumentAttributes] = [ArgumentAttributes(x) for x in ret_attribs_vec_]
        self.rets_bytes_: int = int(rets_bytes_)


@json_data_model
class Buffer:
    def __init__(self, j: Dict[str, Any]) -> None:
        root_id = j["root_id"][0]
        ty = j["type"]

        self.root_id: int = int(root_id)
        self.type: int = int(ty)


@json_data_model
class BufferBinding:
    def __init__(self, j: Dict[str, Any]) -> None:
        binding = j["binding"]
        buffer = j["buffer"]

        self.binding: int = int(binding)
        self.buffer: Buffer = Buffer(buffer)


@json_data_model
class TextureBinding:
    def __init__(self, j: Dict[str, Any]) -> None:
        arg_id = j["arg_id"]
        binding = j["binding"]
        is_storage = j["is_storage"]

        self.arg_id: int = int(arg_id)
        self.binding: int = int(binding)
        self.is_storage: bool = bool(is_storage)


@json_data_model
class RangeBound:
    def __init__(self, j: Dict[str, Any]) -> None:
        self.kind: int = int(j["kind"])
        self.value: int = int(j["value"])


@json_data_model
class RangeForAttributes:
    def __init__(self, j: Dict[str, Any]) -> None:
        self.begin: RangeBound = RangeBound(j["begin"])
        self.end: RangeBound = RangeBound(j["end"])
        self.host_launch: Optional[Dict[str, Any]] = j.get("host_launch")
        self.empty_group_elision: int = j.get("empty_group_elision", 0)


@json_data_model
class TaskAttributes:
    def __init__(self, j: Dict[str, Any]) -> None:
        advisory_num_threads_per_group = j["advisory_num_threads_per_group"]
        advisory_total_num_threads = j["advisory_total_num_threads"]
        buffer_binds = j["buffer_binds"]
        name = j["name"]
        range_for_attribs = j["range_for_attribs"] if "range_for_attribs" in j else None
        task_type = j["task_type"]
        texture_binds = j["texture_binds"]

        self.advisory_num_threads_per_group: int = int(advisory_num_threads_per_group)
        self.advisory_total_num_threads: int = int(advisory_total_num_threads)
        self.buffer_binds: List[BufferBinding] = [BufferBinding(x) for x in buffer_binds]
        self.name: str = str(name)
        self.device_range_dispatch_version: int = int(j.get("device_range_dispatch_version", 0))
        self.range_for_attribs: Optional[RangeForAttributes] = (
            RangeForAttributes(range_for_attribs) if range_for_attribs is not None else None
        )
        self.task_type: int = int(task_type)
        self.texture_binds: List[TextureBinding] = [TextureBinding(x) for x in texture_binds]


@json_data_model
class DeviceCapabilityLevel:
    def __init__(self, j: Dict[str, Any]) -> None:
        key = j["key"]
        value = j["value"]

        self.key: str = str(key)
        self.value: int = int(value)


@json_data_model
class KernelAttributes:
    def __init__(self, j: Dict[str, Any]) -> None:
        ctx_attribs = j["ctx_attribs"]
        is_jit_evaluator = j["is_jit_evaluator"]
        name = j["name"]
        tasks_attribs = j["tasks_attribs"]

        self.ctx_attribs: ContextAttributes = ContextAttributes(ctx_attribs)
        self.is_jit_evaluator: bool = bool(is_jit_evaluator)
        self.name: str = str(name)
        self.tasks_attribs: List[TaskAttributes] = [TaskAttributes(x) for x in tasks_attribs]


@json_data_model
class Metadata:
    def __init__(self, j: Dict[str, Any]) -> None:
        fields = j["fields"]
        kernels = j["kernels"]
        required_caps = j["required_caps"]
        root_buffer_size = j["root_buffer_size"]

        self.fields: List[FieldAttributes] = [FieldAttributes(x) for x in fields]
        self.kernels: List[KernelAttributes] = [KernelAttributes(x) for x in kernels]
        # Schema 2 stores one executable record for direct and Graph aliases.
        # Keep the conventions API's name-based view for header generation.
        names = {kernel.name for kernel in self.kernels}
        for alias in j.get("kernel_aliases", []):
            index, name = alias["kernel_index"], alias["name"]
            if not isinstance(index, int) or not 0 <= index < len(kernels) or not name or name in names:
                raise ValueError("Invalid GFX AOT kernel alias")
            self.kernels.append(KernelAttributes(dict(kernels[index], name=name)))
            names.add(name)
        self.device_range_dispatch = [dict(x, tasks=list(x["tasks"])) for x in j.get("device_range_dispatch", [])]
        self.required_caps: List[DeviceCapabilityLevel] = [DeviceCapabilityLevel(x) for x in required_caps]
        self.root_buffer_size: int = int(root_buffer_size)


def _decode_kernel_rows(document):
    # Header generation consumes the same logical ABI as ordinary module.save().
    # Runtime execution resolves these references in the native module loader.
    pools = {}

    def rows(values, group):
        if not isinstance(values, list):
            raise ValueError("Invalid GFX metadata row array")
        pool = pools.setdefault(group, [])
        result = []
        for value in values:
            if isinstance(value, dict):
                pool.append(value)
            elif type(value) is int and 0 <= value < len(pool):
                value = pool[value]
            else:
                raise ValueError("Invalid GFX metadata row reference")
            result.append(value)
        return result

    version = document["source_metadata_version"]
    if type(version) is not int or not 17 <= version <= 24:
        raise ValueError("Unsupported GFX source metadata version")
    kernels = []
    for kernel in document["kernels"]:
        if type(kernel.get("row_storage")) is not int or kernel["row_storage"] != 1:
            raise ValueError("Invalid GFX kernel row storage")
        ctx = kernel["ctx_attribs"].copy()
        for key in ("arg_attribs_vec_", "ret_attribs_vec_", "arr_access", "grad_arr_access"):
            ctx[key] = rows(ctx[key], key)
        for key in ("args_type_", "rets_type_"):
            if ctx[key] is not None:
                content = ctx[key]["content"]
                ctx[key] = dict(ctx[key], content=dict(content, elements_=rows(content["elements_"], "abi_elements")))
        tasks = [dict(task, **{key: rows(task[key], key)
                               for key in ("buffer_binds", "texture_binds", "acceleration_structure_binds")})
                 for task in kernel["tasks_attribs"]]
        decoded = dict(kernel, ctx_attribs=ctx, tasks_attribs=tasks)
        del decoded["row_storage"]
        kernels.append(decoded)
    result = dict(document, kernels=kernels, metadata_version=version)
    del result["source_metadata_version"]
    return result


def from_json_metadata(j: Dict[str, Any]) -> Metadata:
    if j.get("metadata_version") == 25:
        j = _decode_kernel_rows(j)
    return Metadata(j)


def to_json_metadata(meta_data: Metadata) -> Dict[str, Any]:
    return dump_json_data_model(meta_data)


@json_data_model
class SymbolicArgument:
    def __init__(self, j: Dict[str, Any]) -> None:
        dtype_id = j["dtype_id"]
        element_shape = j["element_shape"]
        field_dim = j["field_dim"]
        name = j["name"]
        num_channels = j["num_channels"]
        tag = j["tag"]

        self.dtype_id: int = int(dtype_id)
        self.element_shape: List[int] = [int(x) for x in element_shape]
        self.field_dim: int = int(field_dim)
        self.name: str = str(name)
        self.num_channels: int = int(num_channels)
        self.tag: int = int(tag)


@json_data_model
class Dispatch:
    def __init__(self, j: Dict[str, Any]) -> None:
        kernel_name = j["kernel_name"]
        symbolic_args = j["symbolic_args"]

        self.kernel_name: str = str(kernel_name)
        self.symbolic_args: List[SymbolicArgument] = [SymbolicArgument(x) for x in symbolic_args]


@json_data_model
class GraphData:
    def __init__(self, j: Dict[str, Any]) -> None:
        dispatches = j["dispatches"]

        self.dispatches = [Dispatch(x) for x in dispatches]


@json_data_model
class Graph:
    def __init__(self, j: Dict[str, Any]) -> None:
        key = j["key"]
        value = j["value"]

        self.key = str(key)
        self.value = GraphData(value)


def from_json_graph(j: Dict[str, Any]) -> Graph:
    return Graph(j)


def to_json_graph(graph: Graph) -> Dict[str, Any]:
    return dump_json_data_model(graph)
