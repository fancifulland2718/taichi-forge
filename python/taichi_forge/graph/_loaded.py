"""Explicit, owner-bound loaded AOT calls in a frontend Graph root sequence."""

from taichi_forge._lib import core
from taichi_forge.lang import impl
from taichi_forge.lang._ndarray import Ndarray, StructNdarray
from taichi_forge.lang.exception import TaichiRuntimeError

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
        values = {}
        try:
            for name, symbol in source.symbols.items():
                value = bindings[name]
                if symbol.tag == core.ArgKind.NDARRAY:
                    if not isinstance(value, Ndarray) or isinstance(value, StructNdarray):
                        raise TypeError("Loaded Graph entries require scalar or tensor JIT ndarrays")
                    if value.arr is None or value._runtime_prog is not source.program:
                        raise TaichiRuntimeError("Loaded Graph storage belongs to a retired or different Program")
                    if (
                        value.dtype != symbol.dtype()
                        or len(value.shape) != symbol.field_dim
                        or tuple(value.element_shape) != tuple(symbol.element_shape)
                    ):
                        raise TaichiRuntimeError(f"Loaded Graph ndarray {name!r} does not match its symbol ABI")
                    key = value._runtime_allocation_identity
                    if key not in imported:
                        imported[key] = runtime.import_ndarray(value)
                    values[name] = imported[key]
                else:
                    if type(value) not in (bool, int, float):
                        raise TypeError("Loaded Graph scalar bindings require bool, int or float")
                    values[name] = value
            # Native preparation validates the artifact's actual ABI (including
            # dtype, rank, tensor shape and gradient requirements), without a
            # Python numerical definition or a launch. Bind aliases only once.
            arguments = [
                (
                    Scalar(str(source.symbols[name].dtype()), values[name])
                    if source.symbols[name].tag == core.ArgKind.SCALAR
                    and index not in source.entry._argument_layout.encoders
                    else values[name]
                )
                for index, name in enumerate(source.arguments)
            ]
            prepared = source.entry.prepare(*arguments)
            prepared._owners = (*prepared._owners, tuple(bindings.values()))
            return _LoadedCall(prepared)
        except BaseException:
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
        if layout.pack_indices:
            raise TaichiRuntimeError("Loaded Graph entries currently accept scalar and ndarray arguments only")
        self.symbols = {}
        self.arguments = tuple(arg.name if isinstance(arg, core.Arg) else None for arg in args)
        for index, arg in enumerate(args):
            if not isinstance(arg, core.Arg) or arg.tag not in (core.ArgKind.SCALAR, core.ArgKind.NDARRAY):
                raise TypeError("dispatch_loaded arguments must be SCALAR or NDARRAY Graph symbols")
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
        return (int(symbol.tag), str(symbol.dtype()), symbol.field_dim, tuple(symbol.element_shape))

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
            RuntimeBinding(name, "ndarray" if arg.tag == core.ArgKind.NDARRAY else "scalar")
            for name, arg in self.symbols.items()
        )

    @property
    def resource_effects(self):
        # Loaded kernels may access module-owned Field roots as well as arrays.
        # Their internal effects are not a frontend purity promise.
        return (
            ResourceEffect(self.module, GraphAccess.OPAQUE, runtime_bound=False),
            *(
                ResourceEffect(name, GraphAccess.READ_WRITE)
                for name, arg in self.symbols.items()
                if arg.tag == core.ArgKind.NDARRAY
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
