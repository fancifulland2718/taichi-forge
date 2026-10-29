"""Cold Field transfers using exported library kernels and instance bindings."""

from contextlib import ExitStack, closing
import json
import math
import operator
import struct
from urllib.parse import quote
from zipfile import ZipFile

from ._plan import ExecutionPlan
from ._runtime import _dtype


def load_transfers(module, contract):
    fixed = {name for name, resource in contract["resources"].items() if resource.get("kind") == "fixed_snode"}
    if set(contract["field_io"]) != fixed or set(contract["field_io_bindings"]) != fixed:
        raise ValueError("Field transfer contract does not cover its fixed resources")
    result = {}
    for name in fixed:
        resource = contract["resources"][name]
        facts, bindings = contract["field_io"][name], contract["field_io_bindings"][name]
        if (set(facts) != {"entries", "offsets"} or set(facts["entries"]) != {"read", "write"}
                or set(bindings) != {"read", "write"}
                or len(facts["offsets"]) != len(resource["shape"])
                or any(type(value) is not int for value in facts["offsets"])):
            raise ValueError("Invalid Field transfer declaration")
        _dtype(resource["dtype"])
        plans = {}
        for operation, entry in bindings.items():
            relative = "component_entries/" + quote(entry, safe="-_.") + ".json"
            if module._source_path.is_dir():
                payload = (module._source_path / relative).read_bytes()
            else:
                with ZipFile(module._source_path) as archive:
                    payload = archive.read(relative)
            metadata = json.loads(payload)
            source = metadata["contract"]
            expected_members = [resource["group"], resource["paths"]]
            if (metadata.get("version") != 1 or metadata.get("entry") != entry
                    or metadata.get("target") != contract["target"]
                    or metadata.get("semantic_key") != facts["entries"][operation]
                    or source.get("captures") or source.get("groups")
                    or not source.get("fixed_members")
                    or any(member != expected_members for member in source["fixed_members"].values())
                    or source.get("fixed_groups") != {resource["group"]: contract["fixed_groups"][resource["group"]]}):
                raise ValueError("Field transfer entry differs from its component resource contract")
            plan = ExecutionPlan([dict(kind="kernel", entry=entry, arguments=["indices", "values"])])
            actual = module._fixed_snode_plan_layouts(plan)
            tree = contract["fixed_bindings"][resource["group"]]
            if actual != {tree: contract["fixed_groups"][resource["group"]]["allocation"]}:
                raise ValueError("Field transfer entry differs from its native compiled tree closure")
            plans[operation] = plan
        result[name] = (tuple(facts["offsets"]), plans)
    return result


class FieldCoordinates:
    """An instance-owned Field, not a dense view of sparse allocator storage.

    Transfers complete synchronously at a cold boundary. Writes activate only
    the addressed cells according to the original SNode semantics; reads do not
    activate cells. Values are contiguous scalar/tensor bytes in coordinate order.
    """

    def __init__(self, instance, resource, transfers):
        self._instance = instance
        self.shape = tuple(resource["shape"])
        self.element_shape = tuple(resource["element_shape"])
        self.dtype = resource["dtype"]
        self._offsets, self._plans = transfers
        self._closed = False

    def _check(self):
        self._instance._check()
        if self._closed:
            raise RuntimeError("Component Field resource is closed")

    def _coordinates(self, coordinates, writing):
        rows = []
        for row in coordinates:
            values = tuple(operator.index(value) for value in row)
            if len(values) != len(self.shape):
                raise ValueError("Field coordinates differ from its index rank")
            if any(value < offset or value >= offset + extent
                   for value, offset, extent in zip(values, self._offsets, self.shape)):
                raise ValueError("Field coordinate is outside its logical shape/offset")
            if any(not -(2**31) <= value < 2**31 for value in values):
                raise ValueError("Field coordinates must fit the existing signed-i32 index ABI")
            rows.append(values)
        if writing and len(set(rows)) != len(rows):
            raise ValueError("Coordinate writes require distinct indices")
        return rows

    def write_at(self, coordinates, values):
        """Upload scalar/tensor bytes at distinct logical coordinates."""
        return self._transfer(coordinates, memoryview(values).tobytes())

    def read_at(self, coordinates):
        """Return scalar/tensor bytes; inactive coordinates produce zero values."""
        return self._transfer(coordinates, None)

    def _transfer(self, coordinates, values):
        instance = self._instance
        runtime = instance._component._module._runtime
        # Keep the instance alive across completion without holding Runtime's
        # submission lock while prepared calls retire their accepted work.
        with instance._close_lock:
            with runtime._lock:
                self._check()
                if instance._has_live_bindings():
                    raise RuntimeError("Field transfers require closing prepared calls at a cold boundary")
                rows = self._coordinates(coordinates, values is not None)
                expected = len(rows) * math.prod(self.element_shape) * _dtype(self.dtype)[2]
                if values is not None and len(values) != expected:
                    raise ValueError("Field coordinate values differ from count or element shape")
                if not rows:
                    return b"" if values is None else None
                instance._transfer_active = True
            try:
                rank = max(len(self.shape), 1)
                index_data = b"".join(struct.pack("=" + "i" * rank, *(row or (0,))) for row in rows)
                with ExitStack() as owners:
                    indices = owners.enter_context(closing(runtime.ndarray("i32", (len(rows), rank))))
                    data = owners.enter_context(closing(runtime.ndarray(self.dtype, len(rows), element_shape=self.element_shape)))
                    indices.write(index_data)
                    if values is not None:
                        data.write(values)
                    domain = (owners.enter_context(runtime.execution_domain(capacity=1, instance_capacity=1))
                              if runtime.arch == "cuda" else None)
                    call = owners.enter_context(instance._component._module.prepare_execution(
                        self._plans["read" if values is None else "write"], dict(indices=indices, values=data),
                        execution_domain=domain, snode_instance=instance._fixed_state))
                    call.initialize()
                    if domain is None:
                        call.run()
                    else:
                        with closing(call.submit()) as ticket:
                            ticket.wait()
                    return data.read() if values is None else None
            finally:
                with runtime._lock:
                    instance._transfer_active = False

    def close(self):
        self._closed = True
