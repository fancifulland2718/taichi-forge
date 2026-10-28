"""Explicit, owner-bound loaded AOT calls in frontend Graph regions."""

import hashlib
from collections.abc import Mapping

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


class _LoadedImports:
    """Cold storage borrowing and value snapshots shared by loaded call kinds."""

    def __init__(self, source, bindings):
        self.source = source
        self.arrays = {}
        self.retained = list(bindings.values())

    def array(self, value, symbol):
        if not isinstance(value, Ndarray) or isinstance(value, StructNdarray):
            raise TypeError("Loaded Graph entries require scalar or tensor JIT ndarrays")
        if value.arr is None or value._runtime_prog is not self.source.program:
            raise TaichiRuntimeError("Loaded Graph storage belongs to a retired or different Program")
        if (
            value.dtype != symbol.dtype()
            or len(value.shape) != symbol.field_dim
            or tuple(value.element_shape) != tuple(symbol.element_shape)
        ):
            raise TaichiRuntimeError(f"Loaded Graph ndarray {symbol.name!r} does not match its symbol ABI")
        key = value._runtime_allocation_identity
        if key not in self.arrays:
            self.arrays[key] = self.source.runtime.import_ndarray(value)
            self.retained.append(value)
        return self.arrays[key]

    @staticmethod
    def plain(value):
        if isinstance(value, Matrix):
            return value.to_list()
        if isinstance(value, Struct):
            return {name: _LoadedImports.plain(member) for name, member in value.entries.items()}
        return value

    def pack(self, value, symbol, gradients, path=()):
        if not isinstance(value, ArgPack):
            raise TypeError("Loaded Graph ARGPACK bindings require a managed ArgPack")
        if value._ArgPack__argpack is None or value._ArgPack__runtime_prog is not self.source.program:
            raise TaichiRuntimeError("Loaded Graph ArgPack belongs to a retired or different Program")
        if value._ArgPack__dtype != symbol.dtype() or tuple(value.keys) != tuple(m.name for m in symbol.members):
            raise TaichiRuntimeError("Loaded Graph ArgPack does not match its symbol ABI")
        result = {}
        for member in symbol.members:
            child = value[member.name]
            member_path = (*path, member.name)
            if member.tag == core.ArgKind.ARGPACK:
                result[member.name] = self.pack(child, member, gradients, member_path)
            elif member.tag == core.ArgKind.NDARRAY:
                result[member.name] = self.array(child, member)
                if child.grad is not None:
                    gradients[member_path] = self.array(child.grad, member)
            else:
                result[member.name] = self.plain(child)
        return result

    def close(self):
        for value in self.arrays.values():
            value.close()


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
        imports = _LoadedImports(source, bindings)
        import_array, pack_values = imports.array, imports.pack
        packs = {}
        values = {}

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
            prepared._owners = (*prepared._owners, tuple(imports.retained))
            return _LoadedCall(prepared)
        except BaseException:
            for value in packs.values():
                value.close()
            imports.close()
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
    def _value_type_matches(symbol, value_type):
        def logical(dtype):
            element = describe_element_type(dtype)
            if element.category == "struct":
                return ("struct", tuple((name, logical(child)) for name, child in core.data_type_struct_members(dtype)))
            return ("tensor" if element.shape else "scalar", str(element.scalar_type), tuple(element.shape))

        def compiled(value):
            if value["kind"] == "struct":
                return ("struct", tuple((member["name"], compiled(member["value"])) for member in value["members"]))
            return (value["kind"], value["dtype"], tuple(value["shape"]))

        return logical(symbol.element_dtype()) == compiled(value_type)

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
            if kind == "value" and not LoadedKernelExecutable._value_type_matches(arg, member["value"]):
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


class _LoadedPlanCall(_LoadedCall):
    def __init__(self, source, prepared):
        super().__init__(prepared)
        self.source = source
        self.initialized = False

    def initialize(self):
        self.source.validate_graph_lifetime()
        if not self.initialized:
            runtime = self.prepared._runtime
            with runtime._lock:
                try:
                    self.prepared.initialize()
                    self.initialized = True
                finally:
                    runtime._pending.discard(self.prepared)

    def __call__(self):
        super().__call__()
        self.initialized = True

    def execution_info(self):
        return dict(
            self.prepared.info,
            kind="loaded_aot_plan",
            plan=self.source.name,
            plan_digest=self.source.digest,
            admission=self.source.execution,
            initialized=self.initialized,
            outer_submission="segmented_prepared_call",
        )


class _LoadedPlanRecording(_LoadedRecording):
    def prepare_graph_execute(self, bindings):
        from taichi_forge_runtime.aot import Scalar

        source = self.executable
        source.validate_graph_lifetime()
        imports = _LoadedImports(source, bindings)
        packs = []
        prepared = None
        try:
            values, gradients = {}, {}
            for name, symbol in source.arguments.items():
                value = bindings[symbol.name]
                if symbol.tag == core.ArgKind.NDARRAY:
                    values[name] = imports.array(value, symbol)
                    if value.grad is not None:
                        gradients[name] = imports.array(value.grad, symbol)
                elif symbol.tag == core.ArgKind.ARGPACK:
                    member_gradients = {}
                    data = imports.pack(value, symbol, member_gradients)
                    packet = source.plan.argpack(source.module, name, data, gradients=member_gradients)
                    packs.append(packet)
                    values[name] = packet
                elif symbol.tag == core.ArgKind.SCALAR:
                    values[name] = Scalar(str(symbol.dtype()), value)
                else:
                    values[name] = source.plan.value(source.module, name, imports.plain(value))
            prepared = source.module.prepare_execution(source.plan, values, gradients=gradients)
            kind = prepared.info["execution_kind"]
            if source.execution == "device" and kind not in ("cuda_graph", "vulkan_commands"):
                raise TaichiRuntimeError(
                    f"Loaded Plan device execution rejected actual {kind}; "
                    "use execution='preserve' only when its native scheduling is intended"
                )
            owners = (tuple(imports.retained), tuple(packs))
            prepared._owners = (*prepared._owners, *owners)
            if prepared._replay_owners is not None:
                prepared._replay_owners = (*prepared._replay_owners, *owners)
            return _LoadedPlanCall(source, prepared)
        except BaseException:
            if prepared is not None:
                prepared.close()
            for packet in packs:
                packet.close()
            imports.close()
            raise


class LoadedPlanExecutable(LoadedKernelExecutable):
    """One module-scoped plan invocation, not flattened application dispatches."""

    def __init__(self, module, name, arguments, *, execution="preserve"):
        from taichi_forge_runtime.aot import Module

        if not isinstance(module, Module):
            raise TypeError("dispatch_loaded_plan requires a loaded AOT Module")
        if execution not in ("preserve", "device"):
            raise ValueError("Loaded Plan execution must be 'preserve' or 'device'")
        self.module, self.runtime = module, module._runtime
        self.program = impl.get_runtime().prog
        if self.program is None or getattr(self.runtime, "_frontend_program", None) is not self.program:
            raise TaichiRuntimeError("dispatch_loaded_plan requires a Runtime borrowed from the current Program")
        self.module_handle = module._handle
        self.validate_graph_lifetime()
        self.plan = module.execution_plan(name)
        self.name, self.execution = name, execution
        self.digest = hashlib.sha256(self.plan.to_json().encode("utf8")).hexdigest()
        if not isinstance(arguments, Mapping) or set(arguments) != set(self.plan.argument_names):
            raise ValueError("Loaded Plan bindings must exactly match its public argument contract")
        self.arguments = dict(arguments)
        self.symbols = {}
        for symbol in self.arguments.values():
            if not isinstance(symbol, core.Arg) or symbol.tag not in (
                core.ArgKind.SCALAR,
                core.ArgKind.NDARRAY,
                core.ArgKind.ARGPACK,
                core.ArgKind.MATRIX,
                core.ArgKind.STRUCT,
            ):
                raise TypeError("Loaded Plan bindings require value, NDARRAY or ARGPACK Graph symbols")
            previous = self.symbols.get(symbol.name)
            if previous is not None and self._symbol_abi(previous) != self._symbol_abi(symbol):
                raise ValueError("Loaded Plan symbols with the same name must have the same ABI")
            self.symbols[symbol.name] = symbol
        # Query the existing plan's actual uses, including initialization and
        # all control regions. Do not interpret conditions or rewrite its code.
        for kernel, uses in self.plan._kernel_argument_uses(module):
            for index, argument in enumerate(uses):
                symbol = self.arguments.get(argument)
                if symbol is None:
                    continue
                layout = kernel._argument_layout
                if (symbol.tag == core.ArgKind.ARGPACK) != (index in layout.pack_indices):
                    raise ValueError("Loaded Plan ArgPack symbol differs from its compiled use")
                if symbol.tag == core.ArgKind.ARGPACK:
                    self._validate_pack(symbol, kernel._get_argpack_layout(index).description, index)
                value_type = layout.value_types.get(index)
                if value_type is not None:
                    if symbol.tag not in (
                        core.ArgKind.SCALAR,
                        core.ArgKind.MATRIX,
                        core.ArgKind.STRUCT,
                    ) or not self._value_type_matches(symbol, value_type):
                        raise ValueError("Loaded Plan value symbol differs from its compiled use")
        self.backend = str(self.runtime.arch)
        if self.backend in ("x64", "arm64"):
            self.backend = "cpu"
        self._action = BackendCommandGraphAction(_LoadedPlanRecording(self))

    def validate_graph_lifetime(self):
        if impl.get_runtime().prog is not self.program:
            raise TaichiRuntimeError("Loaded Plan belongs to a retired Program")
        with self.runtime._lock:
            self.module._check()
            if self.module._handle != self.module_handle:
                raise TaichiRuntimeError("Loaded Plan module generation has changed")

    @property
    def graph_source_contract(self):
        return dict(
            kind="loaded_aot_plan",
            plan=self.name,
            plan_digest=self.digest,
            arguments=[
                dict(parameter=name, symbol=symbol.name, abi=self._symbol_abi(symbol))
                for name, symbol in self.arguments.items()
            ],
            execution=self.execution,
            owner="live_module_generation",
            effects="conservative",
        )

    @property
    def debug_info(self):
        return dict(kind="loaded_aot_plan", plan=self.name, execution=self.execution)
