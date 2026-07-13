import threading
import warnings
from typing import Any, Dict, List

from taichi_forge._lib import core as _ti_core
from taichi_forge.aot.utils import produce_injected_args
from taichi_forge.lang import enums, impl, kernel_impl
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang._texture import Texture
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.matrix import Matrix, MatrixType
from taichi_forge.types.texture_type import FORMAT2TY_CH, TY_CH2FORMAT
from taichi_forge.graph._native import compile_native_graph_node

ArgKind = _ti_core.ArgKind


class _NativeReplayExecutable:
    def __init__(self, nodes):
        self._executables = tuple(node.executable for node in nodes)

    def prewarm(self):
        for executable in self._executables:
            executable.prewarm()
        return self

    def run(self, context):
        for executable in self._executables:
            executable.run()


class _CGraphJITExecutable:
    def __init__(self, compiled_graph):
        self.compiled_graph = compiled_graph
        self._jit_cache = _ti_core.CompiledGraphJITCache()

    def prewarm(self):
        return self

    def run(self, context):
        self.compiled_graph.jit_run_cached(
            context.compile_config(), context.flattened_args(), self._jit_cache
        )

    def invalidate_runtime(self):
        self._jit_cache.clear_runtime_state()

    @property
    def debug_graph_stats(self):
        return self._jit_cache._debug_graph_stats()


class _GraphRunContext:
    _empty_args = {}

    def __init__(self):
        self._args = None
        self._flattened_args = None
        self._compile_config = None
        self._last_arg_signature = None
        self._last_flattened = None

    def begin(self, args):
        self._args = args
        self._flattened_args = None

    def runtime_args(self):
        return self._args

    def compile_config(self):
        if self._compile_config is None:
            self._compile_config = impl.get_runtime().prog.config()
        return self._compile_config

    def flattened_args(self):
        if self._flattened_args is None:
            self._flattened_args = self._flatten_runtime_args(self._args)
        return self._flattened_args

    def _flatten_runtime_args(self, args):
        if not args:
            return self._empty_args

        signature = []
        flattened = {}
        dynamic_items = []
        for k, v in args.items():
            if isinstance(v, Ndarray):
                signature.append((k, "ndarray", id(v), id(v.arr)))
                flattened[k] = v.arr
            elif isinstance(v, Texture):
                signature.append((k, "texture", id(v), id(v.tex)))
                flattened[k] = v.tex
            elif isinstance(v, Matrix):
                signature.append((k, "matrix"))
                dynamic_items.append((k, v.entries))
            elif isinstance(v, (int, float)):
                signature.append((k, "scalar", type(v)))
                dynamic_items.append((k, v))
            else:
                raise TaichiRuntimeError(
                    f"Only python int, float, ti.Matrix and ti.Ndarray are supported as runtime arguments but got {type(v)}"
                )

        signature = tuple(signature)
        if signature == self._last_arg_signature:
            flattened = self._last_flattened
        else:
            self._last_arg_signature = signature
            self._last_flattened = flattened
        for k, v in dynamic_items:
            flattened[k] = v
        return flattened


class _CompiledCGraphNode:
    needs_runtime_args = True

    def __init__(self, compiled_graph, dispatch_count, runtime_arg_names=()):
        self.compiled_graph = compiled_graph
        self.dispatch_count = dispatch_count
        self.runtime_arg_names = frozenset(runtime_arg_names)
        self._jit_cache = _ti_core.CompiledGraphJITCache()

    def run(self, context):
        self.compiled_graph.jit_run_cached(
            context.compile_config(), context.flattened_args(), self._jit_cache
        )

    def invalidate_runtime(self):
        self._jit_cache.clear_runtime_state()

    @property
    def debug_graph_stats(self):
        return self._jit_cache._debug_graph_stats()

    @property
    def debug_info(self):
        return {"kind": "cgraph", "dispatch_count": self.dispatch_count}


class _CompiledNativeGraphNode:
    needs_runtime_args = False
    runtime_arg_names = frozenset()

    def __init__(self, executable):
        self.executable = executable

    def run(self, context):
        self.executable.run()

    @property
    def debug_info(self):
        return self.executable.debug_info


class _GraphSpec:
    def __init__(self, nodes, aot_graph_builder=None, aot_compiled_graph=None):
        self.nodes = tuple(nodes)
        self._aot_graph_builder = aot_graph_builder
        self._aot_compiled_graph = aot_compiled_graph
        self.needs_runtime_args = any(n.needs_runtime_args for n in self.nodes)
        self.dispatch_count = sum(
            getattr(n, "dispatch_count", 0) for n in self.nodes
        )
        self.native_count = sum(
            isinstance(n, _CompiledNativeGraphNode) for n in self.nodes
        )
        self.runtime_arg_names = frozenset().union(
            *(n.runtime_arg_names for n in self.nodes)
        )
        self.repeat_count = 0

    def validate_runtime_args(self, args):
        if not isinstance(args, dict):
            raise TaichiRuntimeError(
                f"Graph.run() expects a dict of runtime arguments, got {type(args)}"
            )
        if args.keys() == self.runtime_arg_names:
            return

        missing = sorted(self.runtime_arg_names.difference(args.keys()))
        unexpected = sorted(args.keys() - self.runtime_arg_names)
        details = []
        if missing:
            details.append(f"Missing graph runtime arguments: {', '.join(missing)}")
        if unexpected:
            details.append(
                f"Unexpected graph runtime arguments: {', '.join(unexpected)}"
            )
        raise TaichiRuntimeError("; ".join(details))

    def instantiate(self, key=None):
        if key is None:
            key = self.instance_key()
        return _GraphInstance(self, key)

    def invalidate_runtime(self):
        for node in self.nodes:
            invalidate = getattr(node, "invalidate_runtime", None)
            if invalidate is not None:
                invalidate()

    def instance_key(self):
        runtime = impl.get_runtime()
        return (impl.runtime_generation(), impl.current_cfg().arch, id(runtime.prog))

    def compiled_graph(self):
        if self.native_count:
            raise TaichiRuntimeError(
                "Graphs containing native nodes cannot be serialized as AOT CGraph yet"
            )
        if self._aot_compiled_graph is None:
            if self._aot_graph_builder is None:
                raise TaichiRuntimeError("This graph does not have an AOT CGraph")
            self._aot_compiled_graph = self._aot_graph_builder.compile()
        return self._aot_compiled_graph

    @property
    def debug_info(self):
        info = {
            "node_count": len(self.nodes),
            "dispatch_count": self.dispatch_count,
            "native_count": self.native_count,
            "repeat_count": self.repeat_count,
            "nodes": [n.debug_info for n in self.nodes],
        }
        if hasattr(self._aot_graph_builder, "item_count"):
            info["aot_item_count"] = self._aot_graph_builder.item_count
        return info


class _GraphExecutable:
    def __init__(self, spec):
        self.spec = spec

    def run(self, args):
        # Runtime arguments and their flattened Python containers belong to
        # one invocation. Reusing this object would let two callers overwrite
        # each other's arguments before the backend cache can serialize its
        # executable state.
        context = _GraphRunContext() if self.spec.needs_runtime_args else None
        if context is not None:
            context.begin(args)
        for node in self.spec.nodes:
            node.run(context)


class _GraphInstance:
    def __init__(self, spec, key):
        self.spec = spec
        self.key = key
        self._executable = None
        self._native_nodes = None
        self._backend_executable = None

        if len(spec.nodes) == 1 and isinstance(spec.nodes[0], _CompiledCGraphNode):
            node = spec.nodes[0]
            self._install_backend_executable(
                _CGraphJITExecutable(node.compiled_graph), "single_cgraph"
            )
        elif not spec.needs_runtime_args:
            self._native_nodes = spec.nodes
            self._kind = "native_only"
            self._set_run_impl(self._run_native_only)
        else:
            self._executable = _GraphExecutable(spec)
            self._kind = "dispatch_loop"
            self._set_run_impl(self._run_general)

        self._maybe_install_native_replay()

    @property
    def run_impl(self):
        return self.run

    def _set_run_impl(self, run_impl):
        # Store an unbound class function. Keeping a bound method here creates
        # a self-cycle (instance -> method -> instance), which can defer a JIT
        # cache and its backend leases until after Program teardown.
        self._run_impl = run_impl.__func__

    def run(self, args):
        self._run_impl(self, args)

    def _maybe_install_native_replay(self):
        arch = impl.current_cfg().arch
        if arch not in (_ti_core.Arch.cuda, _ti_core.Arch.x64, _ti_core.Arch.arm64):
            return
        if self.spec.native_count != len(self.spec.nodes):
            return
        if self.spec.needs_runtime_args:
            return
        kind = "cuda_native_replay" if arch == _ti_core.Arch.cuda else "cpu_native_replay"
        self._install_backend_executable(
            _NativeReplayExecutable(self.spec.nodes),
            kind,
        )

    def _install_backend_executable(self, executable, kind):
        self._backend_executable = executable
        self._kind = kind
        self._set_run_impl(self._run_backend)
        return self

    def invalidate_runtime(self):
        if self._backend_executable is not None:
            invalidate = getattr(
                self._backend_executable, "invalidate_runtime", None
            )
            if invalidate is not None:
                invalidate()

    def prewarm(self):
        if self._backend_executable is not None:
            prewarm = getattr(self._backend_executable, "prewarm", None)
            if prewarm is not None:
                prewarm()
            return self

        for node in self.spec.nodes:
            if isinstance(node, _CompiledNativeGraphNode):
                node.executable.prewarm()
        return self

    def _run_backend(self, args):
        if self._backend_executable is None:
            return self._run_general(args)
        context = _GraphRunContext() if self.spec.needs_runtime_args else None
        if context is not None:
            context.begin(args)
        self._backend_executable.run(context)

    def _run_native_only(self, args):
        for node in self._native_nodes:
            node.run(None)

    def _run_general(self, args):
        self._executable.run(args)

    @property
    def debug_info(self):
        return {"kind": self._kind}

    @property
    def debug_graph_stats(self):
        if isinstance(self._backend_executable, _CGraphJITExecutable):
            return [self._backend_executable.debug_graph_stats]
        return [
            node.debug_graph_stats
            for node in self.spec.nodes
            if isinstance(node, _CompiledCGraphNode)
        ]


class _AOTGraphBuilderPlan:
    def __init__(self):
        self._items = []
        self._runtime_arg_names = set()

    def dispatch(self, kernel_cpp, args):
        self._items.append(("dispatch", kernel_cpp, args))
        self._runtime_arg_names.update(_runtime_arg_names(args))

    @property
    def runtime_arg_names(self):
        """Return symbolic arguments recorded by the durable AOT plan.

        Low-level graph adapters historically dispatched a precompiled kernel
        directly to both ``_aot_graph_plan`` and the native graph builder.  In
        that path ``GraphBuilder.dispatch()`` cannot update its fast-path name
        cache, but the AOT plan still owns the complete symbolic argument list.
        Recovering names here keeps strict runtime validation compatible with
        those adapters without accepting genuinely unknown arguments.
        """
        return self._runtime_arg_names

    def append(self, node):
        # Freeze each append at the point where the runtime builder consumes it.
        # Reusing and then mutating one Sequential between appends must not make
        # the lazily compiled AOT plan observe only its final definition.
        self._items.append(
            ("append", _AOTSequentialSnapshot(node._dispatches), 1)
        )
        self._runtime_arg_names.update(node._runtime_arg_names)

    def snapshot(self):
        items = []
        for item in self._items:
            if item[0] == "dispatch":
                _, kernel_cpp, args = item
                items.append(("dispatch", kernel_cpp, tuple(args)))
            elif item[0] == "append":
                _, node, count = item
                items.append(
                    ("append", _AOTSequentialSnapshot(node._dispatches), count)
                )
            else:
                raise TaichiRuntimeError(f"Unknown AOT graph item kind {item[0]}")

        snapshot = _AOTGraphBuilderPlan()
        snapshot._items = tuple(items)
        snapshot._runtime_arg_names = set(self._runtime_arg_names)
        return snapshot

    def compile(self):
        builder = _ti_core.GraphBuilder()
        for item in self._items:
            if item[0] == "dispatch":
                _, kernel_cpp, args = item
                builder.dispatch(kernel_cpp, args)
            elif item[0] == "append":
                _, node, count = item
                seq = builder.create_sequential()
                node._dispatch_to(seq)
                for _ in range(count):
                    builder.seq().append(seq)
            else:
                raise TaichiRuntimeError(f"Unknown AOT graph item kind {item[0]}")
        return builder.compile()

    @property
    def item_count(self):
        return len(self._items)


def gen_cpp_kernel(kernel_fn, args):
    kernel = kernel_fn._primal
    assert isinstance(kernel, kernel_impl.Kernel)
    injected_args = produce_injected_args(kernel, symbolic_args=args)
    key = kernel.ensure_compiled(*injected_args)
    return kernel.compiled_kernels[key]


def flatten_args(args):
    unzipped_args = []
    # Tuple for matrix args
    # FIXME remove this when native Matrix type is ready
    for arg in args:
        if isinstance(arg, list):
            for sublist in arg:
                unzipped_args.extend(sublist)
        else:
            unzipped_args.append(arg)
    return unzipped_args


def _runtime_arg_names(args):
    return {arg.name for arg in args}


class _AOTSequentialSnapshot:
    def __init__(self, dispatches):
        self._dispatches = tuple(
            (kernel_cpp, tuple(args)) for kernel_cpp, args in dispatches
        )

    def _dispatch_to(self, builder):
        for kernel_cpp, args in self._dispatches:
            builder.dispatch(kernel_cpp, args)


class Sequential:
    def __init__(self):
        self._dispatch_count = 0
        self._dispatches = []
        self._runtime_arg_names = set()

    def dispatch(self, kernel_fn, *args):
        kernel_cpp = gen_cpp_kernel(kernel_fn, args)
        unzipped_args = flatten_args(args)
        self._dispatches.append((kernel_cpp, unzipped_args))
        self._runtime_arg_names.update(_runtime_arg_names(unzipped_args))
        self._dispatch_count += 1

    def _dispatch_to(self, builder):
        for kernel_cpp, args in self._dispatches:
            builder.dispatch(kernel_cpp, args)


class GraphBuilder:
    def __init__(self):
        self._aot_graph_plan = _AOTGraphBuilderPlan()
        self._runtime_graph_builder = _ti_core.GraphBuilder()
        self._dispatch_count = 0
        self._runtime_graph_arg_names = set()
        self._nodes = []

    def dispatch(self, kernel_fn, *args):
        kernel_cpp = gen_cpp_kernel(kernel_fn, args)
        unzipped_args = flatten_args(args)
        self._aot_graph_plan.dispatch(kernel_cpp, unzipped_args)
        self._ensure_runtime_graph_builder().dispatch(kernel_cpp, unzipped_args)
        self._runtime_graph_arg_names.update(_runtime_arg_names(unzipped_args))
        self._dispatch_count += 1

    def create_sequential(self):
        return Sequential()

    def append(self, node):
        # TODO: support appending dispatch node as well.
        assert isinstance(node, Sequential)
        self._aot_graph_plan.append(node)
        node._dispatch_to(self._runtime_graph_builder)
        self._runtime_graph_arg_names.update(node._runtime_arg_names)
        self._dispatch_count += node._dispatch_count

    def _ensure_runtime_graph_builder(self):
        return self._runtime_graph_builder

    def _flush_graph_builder(self):
        if self._dispatch_count == 0:
            return
        # ``_aot_graph_plan`` is the durable source of truth for dispatches.
        # Also consult it here so legacy low-level adapters that dispatch a
        # precompiled kernel directly retain exact runtime-argument validation.
        self._runtime_graph_arg_names.update(
            self._aot_graph_plan.runtime_arg_names
        )
        self._nodes.append(
            _CompiledCGraphNode(
                self._runtime_graph_builder.compile(),
                self._dispatch_count,
                self._runtime_graph_arg_names,
            )
        )
        self._runtime_graph_builder = _ti_core.GraphBuilder()
        self._dispatch_count = 0
        self._runtime_graph_arg_names = set()

    def _append_native(self, node, *, prewarm=False):
        self._flush_graph_builder()
        executable = compile_native_graph_node(node)
        if prewarm:
            executable.prewarm()
        self._nodes.append(_CompiledNativeGraphNode(executable))
        return self

    def append_native(self, node, *, prewarm=False):
        return self._append_native(node, prewarm=prewarm)

    def compile(self):
        self._flush_graph_builder()
        if not self._nodes:
            return Graph(
                _CompiledCGraphNode(
                    self._ensure_runtime_graph_builder().compile(),
                    0,
                    (),
                )
            )
        return Graph(
            _GraphSpec(
                self._nodes,
                aot_graph_builder=self._aot_graph_plan.snapshot(),
            )
        )


class Graph:
    def __init__(self, compiled_graph) -> None:
        self._lifecycle_lock = threading.Lock()
        if isinstance(compiled_graph, _GraphSpec):
            self._spec = compiled_graph
        elif isinstance(compiled_graph, _CompiledCGraphNode):
            self._spec = _GraphSpec(
                [compiled_graph], aot_compiled_graph=compiled_graph.compiled_graph
            )
        else:
            node = _CompiledCGraphNode(compiled_graph, 0, ())
            self._spec = _GraphSpec([node], aot_compiled_graph=compiled_graph)
        self._contains_native_nodes_value = self._spec.native_count > 0
        self._instances = {}
        self._instance = self._instance_for_current_runtime()
        self._runtime_valid = True
        self._run_impl = self._instance.run_impl
        impl.get_runtime().register_runtime_object(self)

    def run(self, args):
        # A graph invocation is one host-side transaction, including mixed
        # CGraph/native sequences. The lock is per Graph and does not wait for
        # GPU completion, so independent graphs remain independently submitable.
        with self._lifecycle_lock:
            self._check_runtime_valid()
            self._spec.validate_runtime_args(args)
            self._run_impl(args)

    def _instance_for_current_runtime(self):
        key = self._spec.instance_key()
        instance = self._instances.get(key)
        if instance is None:
            instance = self._spec.instantiate(key)
            self._instances[key] = instance
        return instance

    def _prewarm(self):
        with self._lifecycle_lock:
            self._check_runtime_valid()
            self._instance.prewarm()
        return self

    def _check_runtime_valid(self):
        if not self._runtime_valid:
            raise TaichiRuntimeError(
                "This graph was compiled before ti.reset() or a runtime "
                "reinitialization. Please rebuild the graph after ti.init()."
            )

    def _invalidate_runtime(self):
        with self._lifecycle_lock:
            self._runtime_valid = False
            self._run_impl = None
            for instance in self._instances.values():
                instance.invalidate_runtime()
            if self._spec is not None:
                self._spec.invalidate_runtime()
            self._instance = None
            self._instances.clear()
            # Definition nodes currently own mixed-graph JIT caches and native
            # executables. Release them before Program/backend teardown so
            # backend allocation leases cannot outlive their Device registry.
            self._spec = None

    @property
    def _debug_info(self):
        with self._lifecycle_lock:
            self._check_runtime_valid()
            return self._spec.debug_info

    @property
    def _instance_debug_info(self):
        with self._lifecycle_lock:
            self._check_runtime_valid()
            return self._instance.debug_info

    @property
    def _graph_stats(self):
        with self._lifecycle_lock:
            self._check_runtime_valid()
            return self._instance.debug_graph_stats

    @property
    def _compiled_graph(self):
        with self._lifecycle_lock:
            self._check_runtime_valid()
            return self._spec.compiled_graph()

    @property
    def _contains_native_nodes(self):
        return self._contains_native_nodes_value


def _deprecate_arg_args(kwargs: Dict[str, Any]):
    if "field_dim" in kwargs:
        warnings.warn(
            "The field_dim argument for ndarray will be deprecated in v1.6.0, use ndim instead.",
            DeprecationWarning,
        )
        if "ndim" in kwargs:
            raise TaichiRuntimeError(
                "field_dim is deprecated, please do not specify field_dim and ndim at the same time."
            )
        kwargs["ndim"] = kwargs["field_dim"]
        del kwargs["field_dim"]
    tag = kwargs["tag"]

    if tag == ArgKind.SCALAR:
        if "element_shape" in kwargs:
            raise TaichiRuntimeError(
                "The element_shape argument for scalar is deprecated in v1.6.0, and is removed in v1.7.0. "
                "Please remove them."
            )

    if tag == ArgKind.NDARRAY:
        if "element_shape" not in kwargs:
            if "dtype" in kwargs:
                dtype = kwargs["dtype"]
                if isinstance(dtype, MatrixType):
                    kwargs["dtype"] = dtype.dtype
                    kwargs["element_shape"] = dtype.get_shape()
                else:
                    kwargs["element_shape"] = ()
        else:
            raise TaichiRuntimeError(
                "The element_shape argument for ndarray is deprecated in v1.6.0, and it is removed in v1.7.0. "
                "Please use vector or matrix data type instead."
            )

    if tag == ArgKind.RWTEXTURE or tag == ArgKind.TEXTURE:
        if "dtype" in kwargs:
            warnings.warn(
                "The dtype argument for texture will be deprecated in v1.6.0, use format instead.",
                DeprecationWarning,
            )
            del kwargs["dtype"]

        if "shape" in kwargs:
            raise TaichiRuntimeError(
                "The shape argument for texture is deprecated in v1.6.0, and it is removed in v1.7.0. "
                "Please use ndim instead. (Note that you no longer need the exact texture size.)"
            )

        if "channel_format" in kwargs or "num_channels" in kwargs:
            if "fmt" in kwargs:
                raise TaichiRuntimeError(
                    "channel_format and num_channels are deprecated, please do not specify channel_format/num_channels and fmt at the same time."
                )
            if tag == ArgKind.RWTEXTURE:
                fmt = TY_CH2FORMAT[(kwargs["channel_format"], kwargs["num_channels"])]
                kwargs["fmt"] = fmt
                raise TaichiRuntimeError(
                    "The channel_format and num_channels arguments for texture are deprecated in v1.6.0, "
                    "and they are removed in v1.7.0. Please use fmt instead."
                )
            else:
                raise TaichiRuntimeError(
                    "The channel_format and num_channels arguments are no longer required for non-RW textures "
                    "since v1.6.0, and they are removed in v1.7.0. Please remove them."
                )


def _check_args(kwargs: Dict[str, Any], allowed_kwargs: List[str]):
    for k, v in kwargs.items():
        if k not in allowed_kwargs:
            raise TaichiRuntimeError(
                f"Invalid argument: {k}, you can only create a graph argument with: {allowed_kwargs}"
            )
        if k == "tag":
            if not isinstance(v, ArgKind):
                raise TaichiRuntimeError(f"tag must be a ArgKind variant, but found {type(v)}.")
        if k == "name":
            if not isinstance(v, str):
                raise TaichiRuntimeError(f"name must be a string, but found {type(v)}.")


def _make_arg_scalar(kwargs: Dict[str, Any]):
    allowed_kwargs = [
        "tag",
        "name",
        "dtype",
    ]
    _check_args(kwargs, allowed_kwargs)
    name = kwargs["name"]
    dtype = kwargs["dtype"]
    if isinstance(dtype, MatrixType):
        raise TaichiRuntimeError(f"Tag ArgKind.SCALAR must specify a scalar type, but found {type(dtype)}.")
    return _ti_core.Arg(ArgKind.SCALAR, name, dtype, 0, [])


def _make_arg_ndarray(kwargs: Dict[str, Any]):
    allowed_kwargs = [
        "tag",
        "name",
        "dtype",
        "ndim",
        "element_shape",
    ]
    _check_args(kwargs, allowed_kwargs)
    name = kwargs["name"]
    ndim = kwargs["ndim"]
    dtype = kwargs["dtype"]
    element_shape = kwargs["element_shape"]
    if isinstance(dtype, MatrixType):
        raise TaichiRuntimeError(f"Tag ArgKind.NDARRAY must specify a scalar type, but found {dtype}.")
    return _ti_core.Arg(ArgKind.NDARRAY, name, dtype, ndim, element_shape)


def _make_arg_matrix(kwargs: Dict[str, Any]):
    allowed_kwargs = [
        "tag",
        "name",
        "dtype",
    ]
    _check_args(kwargs, allowed_kwargs)
    name = kwargs["name"]
    dtype = kwargs["dtype"]
    if not isinstance(dtype, MatrixType):
        raise TaichiRuntimeError(f"Tag ArgKind.MATRIX must specify matrix type, but got {dtype}.")
    return _ti_core.Arg(ArgKind.MATRIX, f"{name}", dtype.dtype, 0, [dtype.n, dtype.m])


def _make_arg_texture(kwargs: Dict[str, Any]):
    allowed_kwargs = [
        "tag",
        "name",
        "ndim",
    ]
    _check_args(kwargs, allowed_kwargs)
    name = kwargs["name"]
    ndim = kwargs["ndim"]
    return _ti_core.Arg(ArgKind.TEXTURE, name, impl.f32, 4, [2] * ndim)


def _make_arg_rwtexture(kwargs: Dict[str, Any]):
    allowed_kwargs = [
        "tag",
        "name",
        "ndim",
        "fmt",
    ]
    _check_args(kwargs, allowed_kwargs)
    name = kwargs["name"]
    ndim = kwargs["ndim"]
    fmt = kwargs["fmt"]
    if fmt == enums.Format.unknown:
        raise TaichiRuntimeError(f"Tag ArgKind.RWTEXTURE must specify a valid color format, but found {fmt}.")
    channel_format, num_channels = FORMAT2TY_CH[fmt]
    return _ti_core.Arg(ArgKind.RWTEXTURE, name, channel_format, num_channels, [2] * ndim)


def _make_arg(kwargs: Dict[str, Any]):
    assert "tag" in kwargs
    _deprecate_arg_args(kwargs)
    proc = {
        ArgKind.SCALAR: _make_arg_scalar,
        ArgKind.NDARRAY: _make_arg_ndarray,
        ArgKind.MATRIX: _make_arg_matrix,
        ArgKind.TEXTURE: _make_arg_texture,
        ArgKind.RWTEXTURE: _make_arg_rwtexture,
    }
    tag = kwargs["tag"]
    return proc[tag](kwargs)


def _kwarg_rewriter(args, kwargs):
    for i, arg in enumerate(args):
        rewrite_map = {
            0: "tag",
            1: "name",
            2: "dtype",
            3: "ndim",
            4: "field_dim",
            5: "element_shape",
            6: "channel_format",
            7: "shape",
            8: "num_channels",
        }
        if i in rewrite_map:
            kwargs[rewrite_map[i]] = arg
        else:
            raise TaichiRuntimeError(f"Unexpected {i}th positional argument")


def Arg(*args, **kwargs):
    _kwarg_rewriter(args, kwargs)
    return _make_arg(kwargs)


__all__ = ["GraphBuilder", "Graph", "Arg", "ArgKind"]
