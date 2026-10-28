"""Explicit, owner-bound loaded AOT calls in a frontend Graph root sequence."""

from taichi_forge._lib import core
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import Ndarray, StructNdarray
from taichi_forge.lang.argpack import ArgPack
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.matrix import Matrix
from taichi_forge.lang.struct import Struct
from taichi_forge.types._argument_descriptor import describe_element_type, describe_symbolic_arg

from ._ir import GraphAccess, ResourceEffect, RuntimeBinding
from ._native import BackendCommandGraphAction, BackendCommandRecording, NativeGraphExecutable


class _LoadedCall:
    def __init__(self, prepared):
        self.prepared = prepared

    def __call__(self):
        runtime = self.prepared._runtime
        with runtime._lock:
            try:
                self.prepared.run()
            finally:
                # This private call is owned by the Graph binding packet and
                # retained through its completion. Ordinary AOT's pending set
                # would otherwise keep every transient frame until Runtime.wait,
                # which Graph completion does not (and should not) call.
                runtime._pending.discard(self.prepared)


class _LoadedRecording(BackendCommandRecording):
    def __init__(self, executable):
        super().__init__(
            backend=executable.backend,
            binding_names=tuple(executable.symbols),
            command_count=1,
        )
        object.__setattr__(self, "executable", executable)

    def prepare_graph_execute(self, bindings):
        from taichi_forge_runtime.aot import Scalar

        source = self.executable
        source.validate_graph_lifetime()
        runtime = source.runtime
        imported = {}
        packs = {}
        retained = list(bindings.values())
        values = {}

        def import_array(value, symbol):
            if not isinstance(value, Ndarray) or isinstance(value, StructNdarray):
                raise TypeError("Loaded Graph entries require scalar or tensor JIT ndarrays")
            if value.arr is None or value._runtime_prog is not source.program:
                raise TaichiRuntimeError("Loaded Graph storage belongs to a retired or different Program")
            if (
                value.dtype != symbol.dtype()
                or len(value.shape) != symbol.field_dim
                or tuple(value.element_shape) != tuple(symbol.element_shape)
            ):
                raise TaichiRuntimeError(f"Loaded Graph ndarray {symbol.name!r} does not match its symbol ABI")
            key = value._runtime_allocation_identity
            if key not in imported:
                imported[key] = runtime.import_ndarray(value)
                retained.append(value)
            return imported[key]

        def plain_value(value):
            if isinstance(value, Matrix):
                return value.to_list()
            if isinstance(value, Struct):
                return {name: plain_value(member) for name, member in value.entries.items()}
            return value

        def pack_values(value, symbol, gradients, path=()):
            if not isinstance(value, ArgPack):
                raise TypeError("Loaded Graph ARGPACK bindings require a managed ArgPack")
            if value._ArgPack__argpack is None or value._ArgPack__runtime_prog is not source.program:
                raise TaichiRuntimeError("Loaded Graph ArgPack belongs to a retired or different Program")
            if value._ArgPack__dtype != symbol.dtype() or tuple(value.keys) != tuple(m.name for m in symbol.members):
                raise TaichiRuntimeError("Loaded Graph ArgPack does not match its symbol ABI")
            result = {}
            for member in symbol.members:
                child = value[member.name]
                member_path = (*path, member.name)
                if member.tag == core.ArgKind.ARGPACK:
                    result[member.name] = pack_values(child, member, gradients, member_path)
                elif member.tag == core.ArgKind.NDARRAY:
                    result[member.name] = import_array(child, member)
                    if child.grad is not None:
                        gradients[member_path] = import_array(child.grad, member)
                else:
                    result[member.name] = plain_value(child)
            return result

        try:
            for name, symbol in source.symbols.items():
                value = bindings[name]
                if symbol.tag == core.ArgKind.NDARRAY:
                    values[name] = import_array(value, symbol)
                elif symbol.tag == core.ArgKind.ARGPACK:
                    gradients = {}
                    values[name] = pack_values(value, symbol, gradients), gradients
                else:
                    if type(value) not in (bool, int, float):
                        raise TypeError("Loaded Graph scalar bindings require bool, int or float")
                    values[name] = value
            # Native preparation validates the artifact's actual ABI (including
            # dtype, rank, tensor shape and gradient requirements), without a
            # Python numerical definition or a launch. Bind aliases only once.
            arguments = []
            argument_gradients = {}
            for index, name in enumerate(source.arguments):
                symbol = source.symbols[name]
                if symbol.tag == core.ArgKind.ARGPACK:
                    if name not in packs:
                        data, gradients = values[name]
                        packs[name] = source.entry.argpack(index, data, gradients=gradients)
                    arguments.append(packs[name])
                elif symbol.tag == core.ArgKind.SCALAR and index not in source.entry._argument_layout.encoders:
                    arguments.append(Scalar(str(symbol.dtype()), values[name]))
                else:
                    arguments.append(values[name])
                    if symbol.tag == core.ArgKind.NDARRAY and bindings[name].grad is not None:
                        argument_gradients[index] = import_array(bindings[name].grad, symbol)
            prepared = (
                source.entry.prepare(*arguments, gradients=argument_gradients)
                if argument_gradients
                else source.entry.prepare(*arguments)
            )
            prepared._owners = (*prepared._owners, tuple(retained))
            return _LoadedCall(prepared)
        except BaseException:
            for value in packs.values():
                value.close()
            for value in imported.values():
                value.close()
            raise

    def execute(self, bindings):
        return self.prepare_graph_execute(bindings)()


class LoadedKernelExecutable(NativeGraphExecutable):
    """An exact entry handle and module generation, never a name lookup at replay."""

    def __init__(self, entry, args):
        from taichi_forge_runtime.aot import Kernel
        from taichi_forge_runtime.aot._runtime import Graph

        if not isinstance(entry, Kernel) or isinstance(entry, Graph):
            raise TypeError("dispatch_loaded expects a loaded AOT Kernel")
        self.entry = entry
        self.module = entry._module
        self.runtime = self.module._runtime
        self.program = impl.get_runtime().prog
        if self.program is None or getattr(self.runtime, "_frontend_program", None) is not self.program:
            raise TaichiRuntimeError("dispatch_loaded requires a Runtime borrowed from the current Program")
        self.module_handle, self.entry_handle = self.module._handle, entry._handle
        self.validate_graph_lifetime()
        if entry._return_layout is not None:
            raise TaichiRuntimeError("Loaded Graph entries must be void; use an output ndarray for results")
        layout = entry._argument_layout
        if len(args) != layout.count:
            raise ValueError("Loaded Graph argument count does not match the exported kernel ABI")
        self.symbols = {}
        self.arguments = tuple(arg.name if isinstance(arg, core.Arg) else None for arg in args)
        for index, arg in enumerate(args):
            if not isinstance(arg, core.Arg) or arg.tag not in (
                core.ArgKind.SCALAR,
                core.ArgKind.NDARRAY,
                core.ArgKind.ARGPACK,
            ):
                raise TypeError("dispatch_loaded arguments must be SCALAR, NDARRAY or ARGPACK Graph symbols")
            if (arg.tag == core.ArgKind.ARGPACK) != (index in layout.pack_indices):
                raise ValueError("Loaded Graph ArgPack symbol does not match the exported kernel ABI")
            if arg.tag == core.ArgKind.ARGPACK:
                self._validate_pack(arg, entry._get_argpack_layout(index).description, index)
            value_type = layout.value_types.get(index)
            if value_type is not None:
                if (
                    arg.tag != core.ArgKind.SCALAR
                    or value_type["kind"] != "scalar"
                    or str(arg.dtype()) != value_type["dtype"]
                ):
                    raise ValueError("Loaded Graph scalar symbol does not match the exported kernel ABI")
            previous = self.symbols.get(arg.name)
            if previous is not None and self._symbol_abi(previous) != self._symbol_abi(arg):
                raise ValueError("Loaded Graph symbols with the same name must have the same ABI")
            self.symbols[arg.name] = arg
        self.backend = str(self.runtime.arch)
        if self.backend in ("x64", "arm64"):
            self.backend = "cpu"
        self._action = BackendCommandGraphAction(_LoadedRecording(self))

    @staticmethod
    def _symbol_abi(symbol):
        return describe_symbolic_arg(symbol).structural_key()

    @staticmethod
    def _validate_pack(symbol, layout, index):
        # Compare logical names and pure-value types. All storage offsets and
        # resource dtype/rank/gradient validation still belong to the native ABI.
        symbols = {}

        def visit(arg, path):
            if arg.tag not in (
                core.ArgKind.ARGPACK,
                core.ArgKind.SCALAR,
                core.ArgKind.MATRIX,
                core.ArgKind.STRUCT,
                core.ArgKind.NDARRAY,
            ):
                raise TaichiRuntimeError("Loaded Graph ArgPack resource kind is not qualified")
            symbols[path] = arg
            if arg.tag == core.ArgKind.ARGPACK:
                for child, member in enumerate(arg.members):
                    visit(member, (*path, child))

        def logical(dtype):
            element = describe_element_type(dtype)
            if element.category == "struct":
                return ("struct", tuple((name, logical(child)) for name, child in core.data_type_struct_members(dtype)))
            return ("tensor" if element.shape else "scalar", str(element.scalar_type), tuple(element.shape))

        def compiled(value):
            if value["kind"] == "struct":
                return ("struct", tuple((member["name"], compiled(member["value"])) for member in value["members"]))
            return (value["kind"], value["dtype"], tuple(value["shape"]))

        visit(symbol, (index,))
        members = layout["members"]
        if set(symbols) != {tuple(member["origin"]["logical"]) for member in members}:
            raise ValueError("Loaded Graph ArgPack members do not match the exported kernel ABI")
        for member in members:
            origin = member["origin"]
            path = tuple(origin["logical"])
            arg = symbols[path]
            kind = {core.ArgKind.ARGPACK: "argpack", core.ArgKind.NDARRAY: "ndarray"}.get(arg.tag, "value")
            if (len(path) > 1 and arg.name != origin["name"]) or kind != origin["kind"]:
                raise ValueError("Loaded Graph ArgPack member does not match the exported kernel ABI")
            if kind == "value" and logical(arg.element_dtype()) != compiled(member["value"]):
                raise ValueError("Loaded Graph ArgPack value type does not match the exported kernel ABI")

    def validate_graph_lifetime(self):
        if impl.get_runtime().prog is not self.program:
            raise TaichiRuntimeError("Loaded Graph entry belongs to a retired Program")
        with self.runtime._lock:
            self.module._check()
            if self.module._handle != self.module_handle or self.entry._handle != self.entry_handle:
                raise TaichiRuntimeError("Loaded Graph entry generation has changed")

    @property
    def recordable_action(self):
        return self._action

    @property
    def runtime_arg_schema(self):
        return tuple(
            RuntimeBinding(
                name, {core.ArgKind.NDARRAY: "ndarray", core.ArgKind.ARGPACK: "argpack"}.get(arg.tag, "scalar")
            )
            for name, arg in self.symbols.items()
        )

    @property
    def graph_binding_symbols(self):
        return tuple(self.symbols.values())

    @property
    def resource_effects(self):
        # Loaded kernels may access module-owned Field roots as well as arrays.
        # Their internal effects are not a frontend purity promise.
        return (
            ResourceEffect(self.module, GraphAccess.OPAQUE, runtime_bound=False),
            *(
                ResourceEffect(name, GraphAccess.READ_WRITE)
                for name, arg in self.symbols.items()
                if arg.tag in (core.ArgKind.NDARRAY, core.ArgKind.ARGPACK)
            ),
        )

    @property
    def graph_source_contract(self):
        return {
            "kind": "loaded_aot_kernel",
            "entry": self.entry._name,
            "arguments": [{"name": name, "abi": self._symbol_abi(self.symbols[name])} for name in self.arguments],
            "owner": "live_module_generation",
            "effects": "conservative",
        }

    @property
    def debug_info(self):
        return {"kind": "loaded_aot_kernel"}
