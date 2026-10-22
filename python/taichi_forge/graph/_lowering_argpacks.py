"""Cold ArgPack values and lifted resources for the existing execution plan ABI."""

from taichi_forge._lib import core
from taichi_forge.lang import impl
from taichi_forge.lang.argpack import ArgPack
from taichi_forge.lang.exception import TaichiRuntimeError


class SourceArgPacks:
    def __init__(self, bindings, fixed_bindings):
        self.owners = {name: value for name, value in fixed_bindings.items() if isinstance(value, ArgPack)}
        self.used = set()
        self._resources = {}
        for name, owner in bindings.items():
            if not isinstance(name, str) or not name or "\0" in name:
                raise ValueError("Source binding names must be nonempty strings without NUL")
            if name in fixed_bindings and fixed_bindings[name] is not owner:
                raise ValueError("Source binding conflicts with private Graph storage")
            self.owners[name] = owner

    def validate_used(self):
        if self.owners.keys() - self.used:
            raise ValueError("source_bindings contains ArgPacks unused by the exported Graph")

    def bind(self, context, symbol, entry, index, jit_source):
        owner = self.owners.get(symbol.name)
        if owner is None or jit_source is None:
            raise TaichiRuntimeError("Fixed Graph ArgPack requires its live owner in source_bindings")
        self.used.add(symbol.name)
        key = (id(owner), entry, index)
        cached = self._resources.get(key)
        if cached is not None:
            return cached
        native, dispatch = jit_source
        packs, resources = {}, {}

        def visit(argument, value, path):
            if argument.tag == core.ArgKind.ARGPACK:
                if (
                    not isinstance(value, ArgPack)
                    or value._ArgPack__argpack is None
                    or value._ArgPack__runtime_prog is not impl.get_runtime().prog
                    or value._ArgPack__dtype != argument.dtype()
                ):
                    raise TaichiRuntimeError("Fixed Graph ArgPack requires a live matching value type")
                children = dict(value.items)
                if list(children) != [member.name for member in argument.members]:
                    raise TaichiRuntimeError("Fixed Graph ArgPack member names differ from its compiled signature")
                packs[path] = value._ArgPack__argpack
                for member_index, member in enumerate(argument.members):
                    visit(member, children[member.name], path + (member_index,))
            elif argument.tag == core.ArgKind.NDARRAY:
                name = context.retained_arrays.require(value)
                context.source_arrays.validate_kernel([argument], [name])
                resources[path] = name
            elif argument.tag not in (core.ArgKind.SCALAR, core.ArgKind.MATRIX, core.ArgKind.STRUCT):
                raise TaichiRuntimeError("Fixed Graph ArgPack currently requires value or ndarray members")

        visit(symbol, owner, (index,))
        _, payloads = native._snapshot_argpack(dispatch, index, packs)
        paths, bindings = [], {}
        for value_index, (path, payload) in enumerate(payloads):
            paths.append(list(path[1:]))
            bindings[f"value_{value_index}"] = context.host_blob_resource(payload)
        resource = context.resource(
            object(),
            "memory",
            "argpack",
            parameters=dict(entry=entry, argument=index, value_count=len(paths), paths=paths),
            bindings=bindings,
        )
        result = resource, resources
        self._resources[key] = result
        return result


def kernel_pack_bindings(context, arguments, entry, jit_source):
    if jit_source is None:
        raise TaichiRuntimeError("Fixed Graph ArgPack requires selected physical dispatch metadata")
    native, dispatch = jit_source
    logical = {}
    for index, argument in enumerate(arguments):
        if argument.tag == core.ArgKind.ARGPACK:
            resource, lifted = context.source_resources.argpacks.bind(context, argument, entry, index, jit_source)
            logical[(index,)] = resource
            logical.update(lifted)
        else:
            name = context.kernel_binding(argument)
            context.source_arrays.validate_kernel([argument], [name])
            logical[(index,)] = name
    physical = {}
    for path, target in native._logical_argument_bindings(dispatch):
        if len(target) == 1:
            physical[target[0]] = logical[tuple(path)]
    if not physical or set(physical) != set(range(len(physical))):
        raise TaichiRuntimeError("Fixed Graph ArgPack has an incomplete physical binding contract")
    return [physical[index] for index in range(len(physical))]
