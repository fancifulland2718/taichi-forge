import ast
import copy
import functools
import inspect
import operator
import os
import re
import sys
import textwrap
import threading
import time
import typing
import types
import warnings
import weakref

import numpy as np
import taichi_forge.lang
from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl, ops, runtime_ops
from taichi_forge.lang.any_array import AnyArray
from taichi_forge.lang._wrap_inspect import getsourcefile, getsourcelines
from taichi_forge.lang._compile_profile import python_compile_profile_event
from taichi_forge.lang.argpack import ArgPackType, ArgPack
from taichi_forge.lang.ast import (
    ASTTransformerContext,
    KernelSimplicityASTChecker,
    transform_tree,
)
from taichi_forge.lang.ast.ast_transformer_utils import ReturnStatus
from taichi_forge.lang.enums import AutodiffMode, Layout
from taichi_forge.lang.exception import (
    TaichiCompilationError,
    TaichiRuntimeError,
    TaichiRuntimeTypeError,
    TaichiSyntaxError,
    TaichiTypeError,
    handle_exception_from_cpp,
)
from taichi_forge.lang.expr import Expr
from taichi_forge.lang.kernel_arguments import KernelArgument
from taichi_forge.lang.matrix import MatrixType
from taichi_forge.lang.shell import _shell_pop_print
from taichi_forge.lang.struct import StructType
from taichi_forge.lang.util import cook_dtype, has_paddle, has_pytorch, to_taichi_type
from taichi_forge.types import (
    ndarray_type,
    primitive_types,
    sparse_matrix_builder,
    template,
    texture_type,
)
from taichi_forge.types.compound_types import CompoundType
from taichi_forge.types._argument_descriptor import describe_annotation
from taichi_forge.types.utils import is_signed

from taichi_forge import _logging

_SOURCE_TEMPLATE_CACHE = os.environ.get("TI_SOURCE_TEMPLATE_CACHE", "1") != "0"


def func(fn, is_real_function=False):
    """Marks a function as callable in Taichi-scope.

    This decorator transforms a Python function into a Taichi one. Taichi
    will JIT compile it into native instructions.

    Args:
        fn (Callable): The Python function to be decorated
        is_real_function (bool): Whether the function is a real function

    Returns:
        Callable: The decorated function

    Example::

        >>> @ti.func
        >>> def foo(x):
        >>>     return x + 2
        >>>
        >>> @ti.kernel
        >>> def run():
        >>>     print(foo(40))  # 42
    """
    is_classfunc = _inside_class(level_of_class_stackframe=3 + is_real_function)

    fun = Func(fn, _classfunc=is_classfunc, is_real_function=is_real_function)

    @functools.wraps(fn)
    def decorated(*args, **kwargs):
        return fun.__call__(*args, **kwargs)

    decorated._is_taichi_function = True
    decorated._is_real_function = is_real_function
    decorated.func = fun
    return decorated


def real_func(fn):
    return func(fn, is_real_function=True)


def pyfunc(fn):
    """Marks a function as callable in both Taichi and Python scopes.

    When called inside the Taichi scope, Taichi will JIT compile it into
    native instructions. Otherwise it will be invoked directly as a
    Python function.

    See also :func:`~taichi_forge.lang.kernel_impl.func`.

    Args:
        fn (Callable): The Python function to be decorated

    Returns:
        Callable: The decorated function
    """
    is_classfunc = _inside_class(level_of_class_stackframe=3)
    fun = Func(fn, _classfunc=is_classfunc, _pyfunc=True)

    @functools.wraps(fn)
    def decorated(*args, **kwargs):
        return fun.__call__(*args, **kwargs)

    decorated._is_taichi_function = True
    decorated._is_real_function = False
    decorated.func = fun
    return decorated


def _get_tree_and_ctx(
    self,
    excluded_parameters=(),
    is_kernel=True,
    arg_features=None,
    args=None,
    ast_builder=None,
    is_real_function=False,
):
    profile_prefix = f"python.frontend.{self.func.__name__}"
    if _SOURCE_TEMPLATE_CACHE:
        with python_compile_profile_event(f"{profile_prefix}.source"):
            cache = getattr(self, "_source_template_cache", None)
            if cache is None:
                file = getsourcefile(self.func)
                src, start_lineno = getsourcelines(self.func)
                src = [textwrap.fill(line, tabsize=4, width=9999) for line in src]
                source = textwrap.dedent("\n".join(src))
                cache = (file, src, start_lineno, source, None)
                self._source_template_cache = cache
            file, src, start_lineno, source, tree_template = cache
        with python_compile_profile_event(f"{profile_prefix}.ast_parse"):
            if tree_template is None:
                tree_template = ast.parse(source)
                tree_template.body[0].decorator_list = []
                self._source_template_cache = (file, src, start_lineno, source, tree_template)
            tree = copy.deepcopy(tree_template)
    else:
        with python_compile_profile_event(f"{profile_prefix}.source"):
            file = getsourcefile(self.func)
            src, start_lineno = getsourcelines(self.func)
            src = [textwrap.fill(line, tabsize=4, width=9999) for line in src]
            source = textwrap.dedent("\n".join(src))
        with python_compile_profile_event(f"{profile_prefix}.ast_parse"):
            tree = ast.parse(source)
        func_body = tree.body[0]
        func_body.decorator_list = []
    with python_compile_profile_event(f"{profile_prefix}.global_vars"):
        global_vars = _get_global_vars(self.func)

    if is_kernel or is_real_function:
        # inject template parameters into globals
        for i in self.template_slot_locations:
            template_var_name = self.arguments[i].name
            global_vars[template_var_name] = args[i]

    with python_compile_profile_event(f"{profile_prefix}.context"):
        ctx = ASTTransformerContext(
            excluded_parameters=excluded_parameters,
            is_kernel=is_kernel,
            func=self,
            arg_features=arg_features,
            global_vars=global_vars,
            argument_data=args,
            src=src,
            start_lineno=start_lineno,
            file=file,
            ast_builder=ast_builder,
            is_real_function=is_real_function,
        )
    return tree, ctx


def _has_explicit_loop_block_dim(tree):
    """Whether the kernel source owns a ti.loop_config(block_dim=...) choice."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Attribute):
            name = node.func.attr
        elif isinstance(node.func, ast.Name):
            name = node.func.id
        if name == "loop_config" and any(
            keyword.arg == "block_dim" for keyword in node.keywords
        ):
            return True
    return False


def _process_args(self, args, kwargs):
    ret = [argument.default for argument in self.arguments]
    len_args = len(args)

    if len_args > len(ret):
        arg_str = ", ".join([str(arg) for arg in args])
        expected_str = ", ".join([f"{arg.name} : {arg.annotation}" for arg in self.arguments])
        msg = f"Too many arguments. Expected ({expected_str}), got ({arg_str})."
        raise TaichiSyntaxError(msg)

    for i, arg in enumerate(args):
        ret[i] = arg

    for key, value in kwargs.items():
        found = False
        for i, arg in enumerate(self.arguments):
            if key == arg.name:
                if i < len_args:
                    raise TaichiSyntaxError(f"Multiple values for argument '{key}'.")
                ret[i] = value
                found = True
                break
        if not found:
            raise TaichiSyntaxError(f"Unexpected argument '{key}'.")

    for i, arg in enumerate(ret):
        if arg is inspect.Parameter.empty:
            if self.arguments[i].annotation is inspect._empty:
                raise TaichiSyntaxError(f"Parameter `{self.arguments[i].name}` missing.")
            else:
                raise TaichiSyntaxError(
                    f"Parameter `{self.arguments[i].name} : {self.arguments[i].annotation}` missing."
                )

    return ret


class Func:
    function_counter = 0

    def __init__(self, _func, _classfunc=False, _pyfunc=False, is_real_function=False):
        self.func = _func
        self.func_id = Func.function_counter
        Func.function_counter += 1
        self.compiled = None
        self.classfunc = _classfunc
        self.pyfunc = _pyfunc
        self.is_real_function = is_real_function
        self.arguments = []
        self.return_type = None
        self.extract_arguments()
        self.template_slot_locations = []
        for i, arg in enumerate(self.arguments):
            if isinstance(arg.annotation, template):
                self.template_slot_locations.append(i)
        self.mapper = TaichiCallableTemplateMapper(self.arguments, self.template_slot_locations)
        self.taichi_functions = {}  # The |Function| class in C++
        self.has_print = False
        self._compiled_program = None
        self._real_function_cache_variants = {}
        self._next_real_function_variant_id = 0
        # P9.A-2 (F2) — set to True once auto_real_function promotes this
        # Func from inline AST expansion to is_real_function=True. Stays True
        # for the rest of the runtime; we never demote back.
        self._auto_promoted = False

    def _ensure_real_function_cache_current_program(self):
        prog = impl.get_runtime().prog
        if self._compiled_program is prog:
            return
        self.compiled = {}
        self.taichi_functions = {}
        self._real_function_cache_variants = {}
        self._next_real_function_variant_id = 0
        self._compiled_program = prog

    def _real_function_compile_variant(self):
        current_kernel = impl.get_runtime().current_kernel
        tier = None
        if current_kernel is not None:
            tier = current_kernel.opt_level
        if tier is None:
            tier = impl.default_cfg().compile_tier
        return (tier,)

    def _real_function_variant_id(self, instance_id):
        variant = (instance_id, self._real_function_compile_variant())
        variant_id = self._real_function_cache_variants.get(variant)
        if variant_id is None:
            variant_id = self._next_real_function_variant_id
            self._next_real_function_variant_id += 1
            self._real_function_cache_variants[variant] = variant_id
        return variant_id

    def _can_auto_promote(self):
        """P9.A-2 (F2) — structural eligibility for auto-promotion.

        Conditions (all must hold):
        - Not already real_function and not pyfunc (pyfunc has dual-scope
          semantics; do not change its IR shape).
        - Active backend is LLVM-based (LLVM is the only codegen with a
          FuncCallStmt visitor).
        - Caller kernel is not in autodiff (real_function has separate
          autodiff handling that we do not auto-engage).
        - All non-template parameters carry an explicit type annotation
          (extract_arguments enforces this for is_real_function=True at
          construction; we re-validate here because the original ctor was
          called with is_real_function=False).
        """
        if self.is_real_function or self.pyfunc:
            return False
        runtime = impl.get_runtime()
        if runtime.current_kernel is None:
            return False
        if runtime.current_kernel.autodiff_mode != AutodiffMode.NONE:
            return False
        if not _ti_core.arch_uses_llvm(impl.default_cfg().arch):
            return False
        for kernel_arg in self.arguments:
            anno = kernel_arg.annotation
            if isinstance(anno, template):
                continue
            if anno is inspect.Parameter.empty:
                return False
        return True

    def _should_auto_promote(self):
        """P9.A-2 (F2) — telemetry-driven trigger.

        Promote once observed inline expansion cost crosses
        cfg.auto_real_function_threshold_us AND we have at least 2 prior
        inline expansions. The 2-call lower bound prevents one-off heavy
        traces from triggering needless C++ codegen for cold paths.
        """
        cfg = impl.default_cfg()
        if not cfg.auto_real_function:
            return False
        runtime = impl.get_runtime()
        stats = runtime._ti_func_expansion_stats.get(self.func_id)
        if stats is None:
            return False
        if stats["call_count"] < 2:
            return False
        threshold_ns = int(cfg.auto_real_function_threshold_us) * 1000
        if stats["cumulative_ns"] < threshold_ns:
            return False
        return True

    def __call__(self, *args, **kwargs):
        args = _process_args(self, args, kwargs)

        if not impl.inside_kernel():
            if not self.pyfunc:
                raise TaichiSyntaxError("Taichi functions cannot be called from Python-scope.")
            return self.func(*args)

        # P9.A-2 (F2) — auto_real_function: one-shot promotion check before
        # entering the inline path. Once promoted, stays promoted.
        if not self.is_real_function and not self._auto_promoted:
            if self._can_auto_promote() and self._should_auto_promote():
                self._auto_promoted = True
                self.is_real_function = True
                stats = impl.get_runtime()._ti_func_expansion_stats.get(self.func_id)
                if stats is not None:
                    stats["promoted"] = True

        if self.is_real_function:
            if impl.get_runtime().current_kernel.autodiff_mode != AutodiffMode.NONE:
                raise TaichiSyntaxError("Real function in gradient kernels unsupported.")
            self._ensure_real_function_cache_current_program()
            instance_id, arg_features = self.mapper.lookup(args)
            variant_id = self._real_function_variant_id(instance_id)
            key = _ti_core.FunctionKey(self.func.__name__, self.func_id, variant_id)
            if self.compiled is None:
                self.compiled = {}
            if key.instance_id not in self.compiled:
                self.do_compile(key=key, args=args, arg_features=arg_features)
            return self.func_call_rvalue(key=key, args=args)
        tree, ctx = _get_tree_and_ctx(
            self,
            is_kernel=False,
            args=args,
            ast_builder=impl.get_runtime().current_kernel.ast_builder(),
            is_real_function=self.is_real_function,
        )
        # P3.b — enforce @ti.func inline recursion depth cap. 0 = disabled.
        # Only non-real @ti.func calls are counted: they perform a Python-level
        # AST expansion that compounds with each nested call, whereas
        # is_real_function uses a C++-side call and caches per-signature IR.
        runtime = impl.get_runtime()
        depth_limit = getattr(runtime, "func_inline_depth_limit", 0)
        runtime.func_inline_depth += 1
        try:
            if depth_limit and runtime.func_inline_depth > depth_limit:
                raise TaichiCompilationError(
                    f"@ti.func inline depth exceeded "
                    f"func_inline_depth_limit={depth_limit}. Either raise the "
                    f"limit via ti.init(func_inline_depth_limit=...) or mark "
                    f"the innermost function with is_real_function=True so "
                    f"the inliner stops at a C++ call site."
                )
            # P9.A-1 (F1) — measure AST inline expansion wall time.
            # Active when profile flag is set or auto_real_function is on
            # (the latter needs the data to drive the F2 heuristic).
            measure = (
                runtime._ti_func_expansion_profile
                or impl.default_cfg().auto_real_function
            )
            with python_compile_profile_event(f"python.func.inline_transform:{self.func.__name__}"):
                if measure:
                    t0 = time.perf_counter_ns()
                    ret = transform_tree(tree, ctx)
                    dt_ns = time.perf_counter_ns() - t0
                    stats = runtime._ti_func_expansion_stats.setdefault(
                        self.func_id,
                        {
                            "name": self.func.__name__,
                            "call_count": 0,
                            "cumulative_ns": 0,
                            "max_ns": 0,
                            "promoted": False,
                        },
                    )
                    stats["call_count"] += 1
                    stats["cumulative_ns"] += dt_ns
                    if dt_ns > stats["max_ns"]:
                        stats["max_ns"] = dt_ns
                else:
                    ret = transform_tree(tree, ctx)
        finally:
            runtime.func_inline_depth -= 1
        if not self.is_real_function:
            if self.return_type and ctx.returned != ReturnStatus.ReturnedValue:
                raise TaichiSyntaxError("Function has a return type but does not have a return statement")
        return ret

    def func_call_rvalue(self, key, args):
        # Skip the template args, e.g., |self|
        assert self.is_real_function
        non_template_args = []
        dbg_info = _ti_core.DebugInfo(impl.get_runtime().get_current_src_info())
        for i, kernel_arg in enumerate(self.arguments):
            anno = kernel_arg.annotation
            if not isinstance(anno, template):
                if id(anno) in primitive_types.type_ids:
                    non_template_args.append(ops.cast(args[i], anno))
                elif isinstance(anno, primitive_types.RefType):
                    non_template_args.append(_ti_core.make_reference(args[i].ptr, dbg_info))
                elif isinstance(anno, ndarray_type.NdarrayType):
                    if not isinstance(args[i], AnyArray):
                        raise TaichiTypeError(
                            f"Expected ndarray in the kernel argument for argument {kernel_arg.name}, got {args[i]}"
                        )
                    non_template_args += _ti_core.get_external_tensor_real_func_args(args[i].ptr, dbg_info)
                else:
                    non_template_args.append(args[i])
        non_template_args = impl.make_expr_group(non_template_args)
        func_call = (
            impl.get_runtime()
            .compiling_callable.ast_builder()
            .insert_func_call(self.taichi_functions[key.instance_id], non_template_args, dbg_info)
        )
        if self.return_type is None:
            return None
        func_call = Expr(func_call)
        ret = []

        for i, return_type in enumerate(self.return_type):
            if id(return_type) in primitive_types.type_ids:
                ret.append(
                    Expr(
                        _ti_core.make_get_element_expr(
                            func_call.ptr, (i,), _ti_core.DebugInfo(impl.get_runtime().get_current_src_info())
                        )
                    )
                )
            elif isinstance(return_type, (StructType, MatrixType)):
                ret.append(return_type.from_taichi_object(func_call, (i,)))
            else:
                raise TaichiTypeError(f"Unsupported return type for return value {i}: {return_type}")
        if len(ret) == 1:
            return ret[0]
        return tuple(ret)

    def do_compile(self, key, args, arg_features):
        tree, ctx = _get_tree_and_ctx(
            self, is_kernel=False, args=args, arg_features=arg_features, is_real_function=self.is_real_function
        )
        fn = impl.get_runtime().prog.create_function(key)

        def func_body():
            old_callable = impl.get_runtime().compiling_callable
            impl.get_runtime().compiling_callable = fn
            ctx.ast_builder = fn.ast_builder()
            with python_compile_profile_event(f"python.func.real_transform:{self.func.__name__}"):
                transform_tree(tree, ctx)
            impl.get_runtime().compiling_callable = old_callable

        self.taichi_functions[key.instance_id] = fn
        self.compiled[key.instance_id] = func_body
        self.taichi_functions[key.instance_id].set_function_body(func_body)

    def extract_arguments(self):
        sig = inspect.signature(self.func)
        if sig.return_annotation not in (inspect.Signature.empty, None):
            self.return_type = sig.return_annotation
            if sys.version_info >= (3, 9):
                if (
                    isinstance(self.return_type, (types.GenericAlias, typing._GenericAlias))
                    and self.return_type.__origin__ is tuple
                ):
                    self.return_type = self.return_type.__args__
            else:
                if isinstance(self.return_type, typing._GenericAlias) and self.return_type.__origin__ is tuple:
                    self.return_type = self.return_type.__args__
            if not isinstance(self.return_type, (list, tuple)):
                self.return_type = (self.return_type,)
            for i, return_type in enumerate(self.return_type):
                if return_type is Ellipsis:
                    raise TaichiSyntaxError("Ellipsis is not supported in return type annotations")
        params = sig.parameters
        arg_names = params.keys()
        for i, arg_name in enumerate(arg_names):
            param = params[arg_name]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                raise TaichiSyntaxError("Taichi functions do not support variable keyword parameters (i.e., **kwargs)")
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                raise TaichiSyntaxError("Taichi functions do not support variable positional parameters (i.e., *args)")
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                raise TaichiSyntaxError("Taichi functions do not support keyword parameters")
            if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise TaichiSyntaxError('Taichi functions only support "positional or keyword" parameters')
            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                if i == 0 and self.classfunc:
                    annotation = template()
                # TODO: pyfunc also need type annotation check when real function is enabled,
                #       but that has to happen at runtime when we know which scope it's called from.
                elif not self.pyfunc and self.is_real_function:
                    raise TaichiSyntaxError(
                        f"Taichi function `{self.func.__name__}` parameter `{arg_name}` must be type annotated"
                    )
            else:
                if isinstance(annotation, ndarray_type.NdarrayType):
                    pass
                elif isinstance(annotation, MatrixType):
                    pass
                elif isinstance(annotation, StructType):
                    pass
                elif id(annotation) in primitive_types.type_ids:
                    pass
                elif isinstance(annotation, template):
                    pass
                elif isinstance(annotation, primitive_types.RefType):
                    pass
                else:
                    raise TaichiSyntaxError(f"Invalid type annotation (argument {i}) of Taichi function: {annotation}")
            self.arguments.append(KernelArgument(annotation, param.name, param.default))


class TaichiCallableTemplateMapper:
    def __init__(self, arguments, template_slot_locations):
        self.arguments = arguments
        self.num_args = len(arguments)
        self.template_slot_locations = template_slot_locations
        self.mapping = {}
        self._next_mapping_id = 0
        self._static_arg_features = tuple("#" for _ in arguments)
        self._dynamic_arg_extractors = tuple(
            (i, arg.annotation, arg.name)
            for i, arg in enumerate(arguments)
            if self._annotation_needs_arg_feature(arg.annotation)
        )
        self._arg_feature_caches = {
            i: weakref.WeakKeyDictionary()
            for i, annotation, _ in self._dynamic_arg_extractors
            if self._annotation_uses_weak_feature_cache(annotation)
        }
        if not self._dynamic_arg_extractors:
            self.mapping[self._static_arg_features] = 0
            self._next_mapping_id = 1

    @staticmethod
    def _annotation_needs_arg_feature(anno):
        return isinstance(
            anno,
            (
                template,
                ArgPackType,
                texture_type.TextureType,
                texture_type.RWTextureType,
                ndarray_type.NdarrayType,
                sparse_matrix_builder,
            ),
        )

    @staticmethod
    def _annotation_uses_weak_feature_cache(anno):
        return isinstance(anno, ndarray_type.NdarrayType)

    @staticmethod
    def _arg_uses_weak_feature_cache(arg):
        if isinstance(
            arg,
            (
                taichi_forge.lang._ndarray.Ndarray,
                taichi_forge.lang._ndarray.StructNdarrayScalarMemberView,
                taichi_forge.lang._ndarray.StructNdarrayTensorMemberView,
            ),
        ):
            return True
        return getattr(arg, "_is_dense_ndarray_view", False)

    @staticmethod
    def _arg_feature_cache_token(arg, anno):
        if anno.needs_grad is None and getattr(arg, "grad", None) is not None:
            return id(arg.grad)
        return 0

    @staticmethod
    def extract_arg(arg, anno, arg_name):
        if isinstance(anno, template):
            # Import lazily: field.py and kernel_impl.py participate in the
            # frontend import cycle.
            from taichi_forge.lang.field import Field  # pylint: disable=C0415

            if isinstance(arg, taichi_forge.lang.snode.SNode):
                return arg.ptr
            if isinstance(arg, taichi_forge.lang.expr.Expr):
                return arg.ptr.get_underlying_ptr_address()
            if isinstance(arg, _ti_core.Expr):
                return arg.get_underlying_ptr_address()
            if isinstance(arg, tuple):
                return tuple(TaichiCallableTemplateMapper.extract_arg(item, anno, arg_name) for item in arg)
            if isinstance(arg, taichi_forge.lang._ndarray.Ndarray):
                raise TaichiRuntimeTypeError(
                    "Ndarray shouldn't be passed in via `ti.template()`, please annotate your kernel using `ti.types.ndarray(...)` instead"
                )

            if (
                isinstance(arg, Field)
                or isinstance(arg, (list, tuple, dict, set))
                or hasattr(arg, "_data_oriented")
            ):
                # [Composite arguments] Return weak reference to the object
                # Taichi kernel will cache the extracted arguments, thus we can't simply return the original argument.
                # Instead, a weak reference to the original value is returned to avoid memory leak.

                # TODO(zhanlue): replacing "tuple(args)" with "hash of argument values"
                # This can resolve the following issues:
                # 1. Invalid weak-ref will leave a dead(dangling) entry in both caches: "self.mapping" and "self.compiled_functions"
                # 2. Different argument instances with same type and same value, will get templatized into seperate kernels.
                return weakref.ref(arg)

            # [Primitive arguments] Return the value
            return arg
        if isinstance(anno, ArgPackType):
            if not isinstance(arg, ArgPack):
                raise TaichiRuntimeTypeError(f"Argument {arg_name} must be a argument pack, got {type(arg)}")
            return tuple(
                TaichiCallableTemplateMapper.extract_arg(arg[name], dtype, arg_name)
                for index, (name, dtype) in enumerate(anno.members.items())
            )
        if isinstance(anno, texture_type.TextureType):
            descriptor = describe_annotation(anno)
            if not isinstance(arg, taichi_forge.lang._texture.Texture):
                raise TaichiRuntimeTypeError(f"Argument {arg_name} must be a texture, got {type(arg)}")
            if arg.num_dims != descriptor.ndim:
                raise TaichiRuntimeTypeError(
                    f"TextureType dimension mismatch for argument {arg_name}: expected {descriptor.ndim}, got {arg.num_dims}"
                )
            return (arg.num_dims,)
        if isinstance(anno, texture_type.RWTextureType):
            descriptor = describe_annotation(anno)
            if not isinstance(arg, taichi_forge.lang._texture.Texture):
                raise TaichiRuntimeTypeError(f"Argument {arg_name} must be a texture, got {type(arg)}")
            if arg.num_dims != descriptor.ndim:
                raise TaichiRuntimeTypeError(
                    f"RWTextureType dimension mismatch for argument {arg_name}: expected {descriptor.ndim}, got {arg.num_dims}"
                )
            if arg.fmt != descriptor.fmt:
                raise TaichiRuntimeTypeError(
                    f"RWTextureType format mismatch for argument {arg_name}: expected {descriptor.fmt}, got {arg.fmt}"
                )
            # (penguinliong) '0' is the assumed LOD level. We currently don't
            # support mip-mapping.
            return arg.num_dims, arg.fmt, 0
        if isinstance(anno, ndarray_type.NdarrayType):
            from taichi_forge.lang._storage_view import DenseNdarrayView  # pylint: disable=C0415

            if isinstance(arg, DenseNdarrayView):
                arg_type = arg.get_type()
                anno.check_matched(arg_type, arg_name)
                needs_grad = False if anno.needs_grad is None else anno.needs_grad
                if needs_grad:
                    raise TaichiRuntimeTypeError(
                        f"Dense storage view argument {arg_name} does not support gradients"
                    )
                return arg_type.element_type, len(arg.shape), False, anno.boundary
            if isinstance(arg, taichi_forge.lang._ndarray.StructNdarrayTensorMemberView):
                anno.check_matched(arg.get_type(), arg_name)
                needs_grad = False if anno.needs_grad is None else anno.needs_grad
                if needs_grad:
                    raise TaichiRuntimeTypeError(
                        f"StructNdarray tensor member view argument {arg_name} does not support gradients"
                    )
                return (
                    arg.scalar_dtype,
                    len(arg.shape),
                    needs_grad,
                    anno.boundary,
                    "struct_tensor_member",
                    arg.offset,
                    arg.stride,
                    arg.dtype,
                )
            if isinstance(arg, taichi_forge.lang._ndarray.StructNdarrayScalarMemberView):
                anno.check_matched(arg.get_type(), arg_name)
                needs_grad = False if anno.needs_grad is None else anno.needs_grad
                if needs_grad:
                    raise TaichiRuntimeTypeError(
                        f"StructNdarray member view argument {arg_name} does not support gradients"
                    )
                return (
                    arg.element_type,
                    len(arg.shape),
                    needs_grad,
                    anno.boundary,
                    "struct_member",
                    arg.offset,
                    arg.stride,
                )
            if isinstance(arg, taichi_forge.lang._ndarray.StructNdarray):
                anno.check_matched(arg.get_type(), arg_name)
                needs_grad = False if anno.needs_grad is None else anno.needs_grad
                if needs_grad:
                    raise TaichiRuntimeTypeError(f"StructNdarray argument {arg_name} does not support gradients")
                return (
                    arg.element_type,
                    len(arg.shape),
                    needs_grad,
                    anno.boundary,
                    "struct_ndarray",
                    arg.struct_type,
                )
            if isinstance(arg, taichi_forge.lang._ndarray.Ndarray):
                anno.check_matched(arg.get_type(), arg_name)
                needs_grad = (arg.grad is not None) if anno.needs_grad is None else anno.needs_grad
                return arg.element_type, len(arg.shape), needs_grad, anno.boundary
            if isinstance(arg, AnyArray):
                ty = arg.get_type()
                anno.check_matched(arg.get_type(), arg_name)
                return ty.element_type, len(arg.shape), ty.needs_grad, anno.boundary
            # external arrays
            shape = getattr(arg, "shape", None)
            if shape is None:
                raise TaichiRuntimeTypeError(f"Invalid type for argument {arg_name}, got {arg}")
            shape = tuple(shape)
            element_shape = ()
            dtype = to_taichi_type(arg.dtype)
            if isinstance(anno.dtype, MatrixType):
                if anno.ndim is not None:
                    if len(shape) != anno.dtype.ndim + anno.ndim:
                        raise ValueError(
                            f"Invalid value for argument {arg_name} - required array has ndim={anno.ndim} element_dim={anno.dtype.ndim}, "
                            f"array with {len(shape)} dimensions is provided"
                        )
                else:
                    if len(shape) < anno.dtype.ndim:
                        raise ValueError(
                            f"Invalid value for argument {arg_name} - required element_dim={anno.dtype.ndim}, "
                            f"array with {len(shape)} dimensions is provided"
                        )
                element_shape = shape[-anno.dtype.ndim :]
                anno_element_shape = anno.dtype.get_shape()
                if None not in anno_element_shape and element_shape != anno_element_shape:
                    raise ValueError(
                        f"Invalid value for argument {arg_name} - required element_shape={anno_element_shape}, "
                        f"array with element shape of {element_shape} is provided"
                    )
            elif anno.dtype is not None:
                # User specified scalar dtype
                if anno.dtype != dtype:
                    raise ValueError(
                        f"Invalid value for argument {arg_name} - required array has dtype={anno.dtype.to_string()}, "
                        f"array with dtype={dtype.to_string()} is provided"
                    )

                if anno.ndim is not None and len(shape) != anno.ndim:
                    raise ValueError(
                        f"Invalid value for argument {arg_name} - required array has ndim={anno.ndim}, "
                        f"array with {len(shape)} dimensions is provided"
                    )
            needs_grad = getattr(arg, "requires_grad", False) if anno.needs_grad is None else anno.needs_grad
            element_type = (
                _ti_core.get_type_factory_instance().get_tensor_type(element_shape, dtype)
                if len(element_shape) != 0
                else arg.dtype
            )
            return element_type, len(shape) - len(element_shape), needs_grad, anno.boundary
        if isinstance(anno, sparse_matrix_builder):
            return arg.dtype
        # Use '#' as a placeholder because other kinds of arguments are not involved in template instantiation
        return "#"

    def extract(self, args):
        return self._extract_features_and_key(args)[0]

    def _extract_features_and_key(self, args):
        if not self._dynamic_arg_extractors:
            return self._static_arg_features, self._static_arg_features
        extracted = list(self._static_arg_features)
        key_features = list(self._static_arg_features)
        for i, annotation, name in self._dynamic_arg_extractors:
            arg = args[i]
            cache = self._arg_feature_caches.get(i)
            if cache is not None and self._arg_uses_weak_feature_cache(arg):
                token = self._arg_feature_cache_token(arg, annotation)
                cached = cache.get(arg)
                if cached is not None and cached[0] == token:
                    extracted[i] = cached[1]
                    key_features[i] = cached[2]
                    continue
                feature = self.extract_arg(arg, annotation, name)
                key_feature = self._make_cache_key(feature)
                cache[arg] = (token, feature, key_feature)
                extracted[i] = feature
                key_features[i] = key_feature
            else:
                feature = self.extract_arg(arg, annotation, name)
                extracted[i] = feature
                key_features[i] = self._make_cache_key(feature)
        return tuple(extracted), tuple(key_features)

    @staticmethod
    def _make_cache_key(value):
        if isinstance(value, tuple):
            return tuple(TaichiCallableTemplateMapper._make_cache_key(item) for item in value)
        if isinstance(value, list):
            return tuple(TaichiCallableTemplateMapper._make_cache_key(item) for item in value)
        if isinstance(value, _ti_core.DataType):
            try:
                hash(value)
                return value
            except RuntimeError:
                return ("DataType", value.to_string())
        return value

    def lookup(self, args):
        if len(args) != self.num_args:
            raise TypeError(f"{self.num_args} argument(s) needed but {len(args)} provided.")

        if not self._dynamic_arg_extractors:
            return 0, self._static_arg_features

        # Field/data-oriented template identities are represented by weak
        # references. Remove dead specialization keys before adding another
        # one so explicit SNodeTree churn does not retain every historical
        # Python Field wrapper. IDs remain monotonic to avoid aliasing a still
        # cached compiled specialization with a newer object.
        dead_keys = [
            key for key in self.mapping if self._cache_key_is_dead(key)
        ]
        for key in dead_keys:
            del self.mapping[key]

        arg_features, key = self._extract_features_and_key(args)
        if key not in self.mapping:
            limit = impl.get_runtime().kernel_specialization_limit
            if len(self.mapping) >= limit:
                raise TaichiRuntimeError(
                    "Callable specialization mapping reached "
                    f"kernel_specialization_limit={limit}. Reuse a finite "
                    "set of template arguments, call ti.reset(), or raise "
                    "the positive limit in ti.init()."
                )
            self.mapping[key] = self._next_mapping_id
            self._next_mapping_id += 1
        return self.mapping[key], arg_features

    @staticmethod
    def _cache_key_is_dead(value):
        if isinstance(value, weakref.ReferenceType):
            return value() is None
        if isinstance(value, tuple):
            return any(
                TaichiCallableTemplateMapper._cache_key_is_dead(item)
                for item in value
            )
        return False


def _get_global_vars(_func):
    # Discussions: https://github.com/taichi-dev/taichi/issues/282
    global_vars = _func.__globals__.copy()

    freevar_names = _func.__code__.co_freevars
    closure = _func.__closure__
    if closure:
        freevar_values = list(map(lambda x: x.cell_contents, closure))
        for name, value in zip(freevar_names, freevar_values):
            global_vars[name] = value

    return global_vars

class Kernel:
    counter = 0

    # P-Compile-6: valid per-kernel compile_tier override values.
    _VALID_OPT_LEVELS = ("fast", "balanced", "full")

    def __init__(self, _func, autodiff_mode, _classkernel=False, opt_level=None):
        self.func = _func
        self.kernel_counter = Kernel.counter
        Kernel.counter += 1
        assert autodiff_mode in (
            AutodiffMode.NONE,
            AutodiffMode.VALIDATION,
            AutodiffMode.FORWARD,
            AutodiffMode.REVERSE,
        )
        self.autodiff_mode = autodiff_mode
        self.grad = None
        self.arguments = []
        self.return_type = None
        self.classkernel = _classkernel
        # P-Compile-6: per-kernel compile_tier override. None = use program
        # default (CompileConfig::compile_tier from ti.init). String value is
        # validated here so users see errors at decoration time, not first
        # invocation.
        if opt_level is not None and opt_level not in Kernel._VALID_OPT_LEVELS:
            raise ValueError(
                f"@ti.kernel(opt_level=...) must be one of {Kernel._VALID_OPT_LEVELS} "
                f"or None, got {opt_level!r}."
            )
        self.opt_level = opt_level
        self.extract_arguments()
        self.template_slot_locations = []
        for i, arg in enumerate(self.arguments):
            if isinstance(arg.annotation, template):
                self.template_slot_locations.append(i)
        self.mapper = TaichiCallableTemplateMapper(self.arguments, self.template_slot_locations)
        impl.get_runtime().kernels.add(self)
        self.reset()
        self.kernel_cpp = None
        self.compiled_kernels = {}
        self.has_print = False

    def ast_builder(self):
        assert self.kernel_cpp is not None
        return self.kernel_cpp.ast_builder()

    def reset(self):
        self.runtime = impl.get_runtime()
        self.compiled_kernels = {}
        self._task_launch_policy_manifests = {}
        self._external_grad_accesses = {}
        self._materializing_external_grad_accesses = set()

    def _mark_external_grad_access(self, arg_indices):
        # Called by AnyArray.grad while the Python AST is being materialized.
        # Compilation is serialized by Runtime._kernel_compilation_lock, so a
        # per-Kernel set is sufficient and adds no steady-launch locking.
        self._materializing_external_grad_accesses.add(arg_indices)

    def extract_arguments(self):
        sig = inspect.signature(self.func)
        if sig.return_annotation not in (inspect._empty, None):
            self.return_type = sig.return_annotation
            if sys.version_info >= (3, 9):
                if (
                    isinstance(self.return_type, (types.GenericAlias, typing._GenericAlias))
                    and self.return_type.__origin__ is tuple
                ):
                    self.return_type = self.return_type.__args__
            else:
                if isinstance(self.return_type, typing._GenericAlias) and self.return_type.__origin__ is tuple:
                    self.return_type = self.return_type.__args__
            if not isinstance(self.return_type, (list, tuple)):
                self.return_type = (self.return_type,)
            for return_type in self.return_type:
                if return_type is Ellipsis:
                    raise TaichiSyntaxError("Ellipsis is not supported in return type annotations")
        params = sig.parameters
        arg_names = params.keys()
        for i, arg_name in enumerate(arg_names):
            param = params[arg_name]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                raise TaichiSyntaxError("Taichi kernels do not support variable keyword parameters (i.e., **kwargs)")
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                raise TaichiSyntaxError("Taichi kernels do not support variable positional parameters (i.e., *args)")
            if param.default is not inspect.Parameter.empty:
                raise TaichiSyntaxError("Taichi kernels do not support default values for arguments")
            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                raise TaichiSyntaxError("Taichi kernels do not support keyword parameters")
            if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise TaichiSyntaxError('Taichi kernels only support "positional or keyword" parameters')
            annotation = param.annotation
            if param.annotation is inspect.Parameter.empty:
                if i == 0 and self.classkernel:  # The |self| parameter
                    annotation = template()
                else:
                    raise TaichiSyntaxError("Taichi kernels parameters must be type annotated")
            else:
                if isinstance(
                    annotation,
                    (
                        template,
                        ndarray_type.NdarrayType,
                        texture_type.TextureType,
                        texture_type.RWTextureType,
                    ),
                ):
                    pass
                elif id(annotation) in primitive_types.type_ids:
                    pass
                elif isinstance(annotation, sparse_matrix_builder):
                    pass
                elif isinstance(annotation, MatrixType):
                    pass
                elif isinstance(annotation, StructType):
                    pass
                elif isinstance(annotation, ArgPackType):
                    pass
                else:
                    raise TaichiSyntaxError(f"Invalid type annotation (argument {i}) of Taichi kernel: {annotation}")
            self.arguments.append(KernelArgument(annotation, param.name, param.default))

    def materialize(
        self, key=None, args=None, arg_features=None, task_launch_policy=None
    ):
        if key is None:
            key = (self.func, 0, self.autodiff_mode)
        self.runtime.materialize()

        if key in self.compiled_kernels:
            return

        # Program owns every C++ Kernel for the whole runtime and compiled
        # Graphs keep stable Kernel pointers, so transparent LRU deletion is
        # unsafe. Bound only the cold cache-miss path; cached launches remain
        # lock-free and continue to work after the budget is reached.
        with self.runtime._kernel_compilation_lock:
            if key in self.compiled_kernels:
                return
            limit = self.runtime.kernel_specialization_limit
            compiled = self.runtime._compiled_specialization_count
            if compiled >= limit:
                raise TaichiRuntimeError(
                    "Runtime compiled specialization budget reached "
                    f"kernel_specialization_limit={limit}. Existing "
                    "specializations remain usable; reuse stable template "
                    "arguments, call ti.reset(), or raise the positive limit "
                    "in ti.init()."
                )
            self._materialize_uncached(key, args, arg_features, task_launch_policy)
            self.runtime._compiled_specialization_count += 1

    def _materialize_uncached(self, key, args, arg_features, task_launch_policy=None):
        kernel_name = f"{self.func.__name__}_c{self.kernel_counter}_{key[1]}"
        _logging.trace(f"Compiling kernel {kernel_name} in {self.autodiff_mode}...")

        tree, ctx = _get_tree_and_ctx(
            self,
            args=args,
            excluded_parameters=self.template_slot_locations,
            arg_features=arg_features,
        )

        if self.autodiff_mode != AutodiffMode.NONE:
            KernelSimplicityASTChecker(self.func).visit(tree)

        task_launch_policy_injected = False

        # Do not change the name of 'taichi_ast_generator'
        # The warning system needs this identifier to remove unnecessary messages
        def taichi_ast_generator(kernel_cxx):
            nonlocal task_launch_policy_injected
            if self.runtime.inside_kernel:
                raise TaichiSyntaxError(
                    "Kernels cannot call other kernels. I.e., nested kernels are not allowed. "
                    "Please check if you have direct/indirect invocation of kernels within kernels. "
                    "Note that some methods provided by the Taichi standard library may invoke kernels, "
                    "and please move their invocations to Python-scope."
                )
            self.kernel_cpp = kernel_cxx
            self.runtime.inside_kernel = True
            self.runtime.current_kernel = self
            assert self.runtime.compiling_callable is None
            self.runtime.compiling_callable = kernel_cxx
            try:
                ctx.ast_builder = kernel_cxx.ast_builder()
                if (
                    task_launch_policy is not None
                    and task_launch_policy.mode != "auto"
                    and not _has_explicit_loop_block_dim(tree)
                ):
                    # Use the same frontend loop decorator as source-level
                    # ti.loop_config. A late FrontendForStmt mutation can keep
                    # constant range setup in an extra serial offload.
                    ctx.ast_builder.block_dim(task_launch_policy.block_dim)
                    task_launch_policy_injected = True
                with python_compile_profile_event(
                    f"python.kernel.ast_transform:{self.func.__name__}"
                ):
                    transform_tree(tree, ctx)
                if not ctx.is_real_function:
                    if self.return_type and ctx.returned != ReturnStatus.ReturnedValue:
                        raise TaichiSyntaxError("Kernel has a return type but does not have a return statement")
            finally:
                self.runtime.inside_kernel = False
                self.runtime.current_kernel = None
                self.runtime.compiling_callable = None

        self._materializing_external_grad_accesses.clear()
        try:
            taichi_kernel = impl.get_runtime().prog.create_kernel(
                taichi_ast_generator, kernel_name, self.autodiff_mode
            )
        except Exception:
            self._materializing_external_grad_accesses.clear()
            raise
        if self._materializing_external_grad_accesses:
            self._external_grad_accesses[key] = frozenset(
                self._materializing_external_grad_accesses
            )
        self._materializing_external_grad_accesses.clear()
        # P-Compile-6: apply per-kernel compile_tier override (if set on the
        # decorator). Stored on the C++ Kernel; consumed in
        # Program::compile_kernel by copying CompileConfig and overriding the
        # tier. Cache key already includes compile_tier so cache entries are
        # auto-segregated.
        if self.opt_level is not None:
            taichi_kernel.set_compile_tier_override(self.opt_level)
        if task_launch_policy is not None and task_launch_policy.mode != "auto":
            taichi_kernel.set_task_launch_policy(
                task_launch_policy.mode,
                task_launch_policy.block_dim,
                task_launch_policy_injected,
            )
        assert key not in self.compiled_kernels
        self.compiled_kernels[key] = taichi_kernel

    def launch_kernel(
        self,
        t_kernel,
        *args,
        _allocate_all_external_grad=False,
        _explicit_external_grad_args=frozenset(),
    ):
        assert len(args) == len(self.arguments), f"{len(self.arguments)} arguments needed but {len(args)} provided"

        tmps = []
        callbacks = []

        actual_argument_slot = 0
        launch_ctx = t_kernel.make_launch_context()
        max_arg_num = 64
        exceed_max_arg_num = False

        def set_arg_ndarray(indices, v):
            v_primal = v.arr
            v_grad = v.grad.arr if v.grad else None
            if v_primal is None:
                raise TaichiRuntimeError(
                    "Cannot submit an Ndarray after its Taichi runtime has been reset"
                )
            if v.grad is not None and v_grad is None:
                raise TaichiRuntimeError(
                    "Cannot submit an Ndarray gradient after its Taichi runtime has been reset"
                )
            if v_grad is None:
                launch_ctx.set_arg_ndarray(indices, v_primal)
            else:
                launch_ctx.set_arg_ndarray_with_grad(indices, v_primal, v_grad)

        def set_arg_texture(indices, v):
            if v.tex is None:
                raise TaichiRuntimeError(
                    "Cannot submit a Texture after its Taichi runtime has been reset"
                )
            launch_ctx.set_arg_texture(indices, v.tex)

        def set_arg_rw_texture(indices, v):
            if v.tex is None:
                raise TaichiRuntimeError(
                    "Cannot submit a Texture after its Taichi runtime has been reset"
                )
            launch_ctx.set_arg_rw_texture(indices, v.tex)

        def set_arg_ext_array(indices, v, needed):
            # Element shapes are already specialized in Taichi codegen.
            # The shape information for element dims are no longer needed.
            # Therefore we strip the element shapes from the shape vector,
            # so that it only holds "real" array shapes.
            is_soa = needed.layout == Layout.SOA
            array_shape = v.shape
            if functools.reduce(operator.mul, array_shape, 1) > np.iinfo(np.int32).max:
                warnings.warn("Ndarray index might be out of int32 boundary but int64 indexing is not supported yet.")
            if needed.dtype is None or id(needed.dtype) in primitive_types.type_ids:
                element_dim = 0
            else:
                element_dim = needed.dtype.ndim
                array_shape = v.shape[element_dim:] if is_soa else v.shape[:-element_dim]
            needs_grad_pointer = (
                _allocate_all_external_grad
                and getattr(v, "requires_grad", False)
            ) or indices in _explicit_external_grad_args
            if not needs_grad_pointer and getattr(v, "grad", None) is None:
                from taichi_forge.interop._dlpack import (  # pylint: disable=C0415
                    _adapt_external_array,
                )

                element_shape = ()
                if isinstance(needed.dtype, MatrixType):
                    element_shape = tuple(needed.dtype.get_shape())
                external_view = _adapt_external_array(
                    v,
                    element_shape=element_shape,
                    layout="soa" if is_soa else "aos",
                )
                if external_view is not None:
                    tmps.append(external_view)
                    launch_ctx.set_arg_runtime_storage(
                        indices, external_view.runtime_argument
                    )
                    return
            if isinstance(v, np.ndarray):
                if v.flags.c_contiguous:
                    launch_ctx.set_arg_external_array_with_shape(indices, int(v.ctypes.data), v.nbytes, array_shape, 0)
                elif v.flags.f_contiguous:
                    # TODO: A better way that avoids copying is saving strides info.
                    tmp = np.ascontiguousarray(v)
                    # Purpose: DO NOT GC |tmp|!
                    tmps.append(tmp)

                    def callback(original, updated):
                        np.copyto(original, np.asfortranarray(updated))

                    callbacks.append(functools.partial(callback, v, tmp))
                    launch_ctx.set_arg_external_array_with_shape(
                        indices, int(tmp.ctypes.data), tmp.nbytes, array_shape, 0
                    )
                else:
                    raise ValueError(
                        "Non contiguous numpy arrays are not supported, please call np.ascontiguousarray(arr) "
                        "before passing it into taichi kernel."
                    )
            elif has_pytorch():
                import torch  # pylint: disable=C0415

                if isinstance(v, torch.Tensor):
                    if not v.is_contiguous():
                        raise ValueError(
                            "Non contiguous tensors are not supported, please call tensor.contiguous() before "
                            "passing it into taichi kernel."
                        )
                    taichi_arch = self.runtime.prog.config().arch

                    def get_call_back(u, v):
                        def call_back():
                            with torch.no_grad():
                                u.copy_(v)

                        return call_back

                    needs_grad_pointer = (
                        _allocate_all_external_grad and v.requires_grad
                    ) or indices in _explicit_external_grad_args
                    if needs_grad_pointer and v.grad is None:
                        if not v.requires_grad:
                            raise ValueError(
                                "Kernel explicitly accesses a Torch tensor gradient, "
                                "but the tensor has requires_grad=False and no .grad tensor."
                            )
                        v.grad = torch.zeros_like(v)

                    if v.grad is not None:
                        if not isinstance(v.grad, torch.Tensor):
                            raise ValueError(
                                f"Expecting torch.Tensor for gradient tensor, but getting {v.grad.__class__.__name__} instead"
                            )
                        if not v.grad.is_contiguous():
                            raise ValueError(
                                "Non contiguous gradient tensors are not supported, please call tensor.grad.contiguous() before passing it into taichi kernel."
                            )
                        if v.grad.shape != v.shape:
                            raise ValueError(
                                "Gradient tensor shape must match its primal tensor: "
                                f"got grad shape {tuple(v.grad.shape)} and primal shape {tuple(v.shape)}"
                            )
                        if v.grad.dtype != v.dtype:
                            raise ValueError(
                                "Gradient tensor dtype must match its primal tensor: "
                                f"got grad dtype {v.grad.dtype} and primal dtype {v.dtype}"
                            )
                        if v.grad.device != v.device:
                            raise ValueError(
                                "Gradient tensor device must match its primal tensor: "
                                f"got grad device {v.grad.device} and primal device {v.device}"
                            )

                    tmp = v
                    tmp_grad = v.grad
                    if (str(v.device) != "cpu") and not (
                        str(v.device).startswith("cuda") and taichi_arch == _ti_core.Arch.cuda
                    ):
                        # Getting a torch CUDA tensor on Taichi non-cuda arch:
                        # We just replace it with a CPU tensor and by the end of kernel execution we'll use the
                        # callback to copy the values back to the original CUDA tensor.
                        host_v = v.to(device="cpu", copy=True)
                        tmp = host_v
                        callbacks.append(get_call_back(v, host_v))
                        if v.grad is not None:
                            host_grad = v.grad.to(device="cpu", copy=True)
                            tmp_grad = host_grad
                            callbacks.append(get_call_back(v.grad, host_grad))

                    launch_ctx.set_arg_external_array_with_shape(
                        indices,
                        int(tmp.data_ptr()),
                        tmp.element_size() * tmp.nelement(),
                        array_shape,
                        int(tmp_grad.data_ptr()) if tmp_grad is not None else 0,
                    )
                else:
                    raise TaichiRuntimeTypeError(
                        f"Argument {needed.to_string()} cannot be converted into required type {v}"
                    )
            elif has_paddle():
                import paddle  # pylint: disable=C0415

                if isinstance(v, paddle.Tensor):
                    # For now, paddle.fluid.core.Tensor._ptr() is only available on develop branch
                    def get_call_back(u, v):
                        def call_back():
                            u.copy_(v, False)

                        return call_back

                    tmp = v.value().get_tensor()
                    taichi_arch = self.runtime.prog.config().arch
                    if v.place.is_gpu_place():
                        if taichi_arch != _ti_core.Arch.cuda:
                            # Paddle cuda tensor on Taichi non-cuda arch
                            host_v = v.cpu()
                            tmp = host_v.value().get_tensor()
                            callbacks.append(get_call_back(v, host_v))
                    elif v.place.is_cpu_place():
                        if taichi_arch == _ti_core.Arch.cuda:
                            # Paddle cpu tensor on Taichi cuda arch
                            gpu_v = v.cuda()
                            tmp = gpu_v.value().get_tensor()
                            callbacks.append(get_call_back(v, gpu_v))
                    else:
                        # Paddle do support many other backends like XPU, NPU, MLU, IPU
                        raise TaichiRuntimeTypeError(f"Taichi do not support backend {v.place} that Paddle support")
                    launch_ctx.set_arg_external_array_with_shape(
                        indices, int(tmp._ptr()), v.element_size() * v.size, array_shape, 0
                    )
                else:
                    raise TaichiRuntimeTypeError(
                        f"Argument {needed.to_string()} cannot be converted into required type {v}"
                    )
            else:
                raise TaichiRuntimeTypeError(
                    f"Argument {needed.to_string()} cannot be converted into required type {v}"
                )

        def set_arg_matrix(indices, v, needed):
            if needed.dtype in primitive_types.real_types:

                def cast_func(x):
                    if not isinstance(x, (int, float, np.integer, np.floating)):
                        raise TaichiRuntimeTypeError(
                            f"Argument {needed.dtype.to_string()} cannot be converted into required type {type(x)}"
                        )
                    return float(x)

            elif needed.dtype in primitive_types.integer_types:

                def cast_func(x):
                    if not isinstance(x, (int, np.integer)):
                        raise TaichiRuntimeTypeError(
                            f"Argument {needed.dtype.to_string()} cannot be converted into required type {type(x)}"
                        )
                    return int(x)

            else:
                raise ValueError(f"Matrix dtype {needed.dtype} is not integer type or real type.")

            if needed.ndim == 2:
                v = [cast_func(v[i, j]) for i in range(needed.n) for j in range(needed.m)]
            else:
                v = [cast_func(v[i]) for i in range(needed.n)]
            v = needed(*v)
            needed.set_kernel_struct_args(v, launch_ctx, indices)

        def set_arg_sparse_matrix_builder(indices, v):
            if self.runtime.prog.config().arch == _ti_core.Arch.vulkan:
                launch_ctx.set_arg_ndarray(indices, v._get_ndarray())
            else:
                # LLVM backends consume the builder through a raw pointer.
                launch_ctx.set_arg_uint(indices, v._get_ndarray_addr())

        def set_arg_struct_member_ndarray(indices, v):
            launch_ctx.set_arg_ndarray(indices, v.base.arr)

        def set_arg_dense_storage(indices, v):
            launch_ctx.set_arg_runtime_storage(indices, v.runtime_argument)

        set_later_list = []

        def recursive_set_args(needed, provided, v, indices):
            in_argpack = len(indices) > 1
            nonlocal actual_argument_slot, exceed_max_arg_num, set_later_list
            if actual_argument_slot >= max_arg_num:
                exceed_max_arg_num = True
                return 0
            actual_argument_slot += 1
            if isinstance(needed, ArgPackType):
                if not isinstance(v, ArgPack):
                    raise TaichiRuntimeTypeError.get(indices, str(needed), str(provided))
                tmps.append(v)
                idx_new = 0
                for j, (name, anno) in enumerate(needed.members.items()):
                    idx_new += recursive_set_args(anno, type(v[name]), v[name], indices + (idx_new,))
                native_argpack = v._ArgPack__argpack
                if native_argpack is None:
                    raise TaichiRuntimeError(
                        "Cannot submit an ArgPack after its Taichi runtime has been reset"
                    )
                launch_ctx.set_arg_argpack(indices, native_argpack)
                return 1
            # Note: do not use sth like "needed == f32". That would be slow.
            if id(needed) in primitive_types.real_type_ids:
                if not isinstance(v, (float, int, np.floating, np.integer)):
                    raise TaichiRuntimeTypeError.get(indices, needed.to_string(), provided)
                if in_argpack:
                    return 1
                launch_ctx.set_arg_float(indices, float(v))
                return 1
            if id(needed) in primitive_types.integer_type_ids:
                if not isinstance(v, (int, np.integer)):
                    raise TaichiRuntimeTypeError.get(indices, needed.to_string(), provided)
                if in_argpack:
                    return 1
                if is_signed(cook_dtype(needed)):
                    launch_ctx.set_arg_int(indices, int(v))
                else:
                    launch_ctx.set_arg_uint(indices, int(v))
                return 1
            if isinstance(needed, sparse_matrix_builder):
                if in_argpack:
                    set_later_list.append((set_arg_sparse_matrix_builder, (v,)))
                    return 0
                set_arg_sparse_matrix_builder(indices, v)
                return 1
            if isinstance(needed, ndarray_type.NdarrayType) and isinstance(
                v, taichi_forge.lang._ndarray.StructNdarrayTensorMemberView
            ):
                if in_argpack:
                    set_later_list.append((set_arg_struct_member_ndarray, (v,)))
                    return 0
                set_arg_struct_member_ndarray(indices, v)
                return 1
            if isinstance(needed, ndarray_type.NdarrayType) and isinstance(
                v, taichi_forge.lang._ndarray.StructNdarrayScalarMemberView
            ):
                if in_argpack:
                    set_later_list.append((set_arg_struct_member_ndarray, (v,)))
                    return 0
                set_arg_struct_member_ndarray(indices, v)
                return 1
            if isinstance(needed, ndarray_type.NdarrayType):
                from taichi_forge.lang._storage_view import DenseNdarrayView  # pylint: disable=C0415

                if isinstance(v, DenseNdarrayView):
                    if in_argpack:
                        raise TaichiRuntimeTypeError(
                            "Dense storage views are not supported inside ArgPack"
                        )
                    set_arg_dense_storage(indices, v)
                    return 1
            if isinstance(needed, ndarray_type.NdarrayType) and isinstance(v, taichi_forge.lang._ndarray.Ndarray):
                if in_argpack:
                    set_later_list.append((set_arg_ndarray, (v,)))
                    return 0
                set_arg_ndarray(indices, v)
                return 1
            if isinstance(needed, texture_type.TextureType) and isinstance(v, taichi_forge.lang._texture.Texture):
                if in_argpack:
                    set_later_list.append((set_arg_texture, (v,)))
                    return 0
                set_arg_texture(indices, v)
                return 1
            if isinstance(needed, texture_type.RWTextureType) and isinstance(v, taichi_forge.lang._texture.Texture):
                if in_argpack:
                    set_later_list.append((set_arg_rw_texture, (v,)))
                    return 0
                set_arg_rw_texture(indices, v)
                return 1
            if isinstance(needed, ndarray_type.NdarrayType):
                if in_argpack:
                    set_later_list.append((set_arg_ext_array, (v, needed)))
                    return 0
                set_arg_ext_array(indices, v, needed)
                return 1
            if isinstance(needed, MatrixType):
                if in_argpack:
                    return 1
                set_arg_matrix(indices, v, needed)
                return 1
            if isinstance(needed, StructType):
                if in_argpack:
                    return 1
                if not isinstance(v, needed):
                    raise TaichiRuntimeTypeError(f"Argument {provided} cannot be converted into required type {needed}")
                needed.set_kernel_struct_args(v, launch_ctx, indices)
                return 1
            raise ValueError(f"Argument type mismatch. Expecting {needed}, got {type(v)}.")

        template_num = 0
        for i, val in enumerate(args):
            needed_ = self.arguments[i].annotation
            if isinstance(needed_, template):
                template_num += 1
                continue
            recursive_set_args(needed_, type(val), val, (i - template_num,))

        for i, (set_arg_func, params) in enumerate(set_later_list):
            set_arg_func((len(args) - template_num + i,), *params)

        if exceed_max_arg_num:
            raise TaichiRuntimeError(
                f"The number of elements in kernel arguments is too big! Do not exceed {max_arg_num} on {_ti_core.arch_name(impl.current_cfg().arch)} backend."
            )

        try:
            prog = impl.get_runtime().prog
            # Compile/cache lookup and launch form one native SNodeTree
            # lifecycle transaction. This also removes a cross-language call
            # from the steady path while preventing explicit tree destruction
            # from retiring the compiled handle between the two operations.
            prog.compile_and_launch_kernel(
                prog.config(), prog.get_device_caps(), t_kernel, launch_ctx
            )
        except Exception as e:
            e = handle_exception_from_cpp(e)
            if impl.get_runtime().print_full_traceback:
                raise e
            raise e from None

        ret = None
        ret_dt = self.return_type
        has_ret = ret_dt is not None

        if has_ret or self.has_print:
            runtime_ops.sync()

        if has_ret:
            ret = []
            for i, ret_type in enumerate(ret_dt):
                ret.append(self.construct_kernel_ret(launch_ctx, ret_type, (i,)))
            if len(ret_dt) == 1:
                ret = ret[0]
        if callbacks:
            for c in callbacks:
                c()

        return ret

    def construct_kernel_ret(self, launch_ctx, ret_type, index=()):
        if isinstance(ret_type, CompoundType):
            return ret_type.from_kernel_struct_ret(launch_ctx, index)
        if ret_type in primitive_types.integer_types:
            if is_signed(cook_dtype(ret_type)):
                return launch_ctx.get_struct_ret_int(index)
            return launch_ctx.get_struct_ret_uint(index)
        if ret_type in primitive_types.real_types:
            return launch_ctx.get_struct_ret_float(index)
        raise TaichiRuntimeTypeError(f"Invalid return type on index={index}")

    def ensure_compiled(self, *args):
        with python_compile_profile_event(f"python.kernel.ensure_compiled:{self.func.__name__}"):
            instance_id, arg_features = self.mapper.lookup(args)
            key = (self.func, instance_id, self.autodiff_mode)
            self.materialize(key=key, args=args, arg_features=arg_features)
            return key

    def _ensure_compiled_with_task_launch_policy(self, policy, *args):
        with python_compile_profile_event(
            f"python.kernel.ensure_compiled_with_task_launch_policy:{self.func.__name__}"
        ):
            instance_id, arg_features = self.mapper.lookup(args)
            key = (
                self.func,
                instance_id,
                self.autodiff_mode,
                policy._specialization_key,
            )
            if (
                key not in self._task_launch_policy_manifests
                and threading.current_thread() is not threading.main_thread()
            ):
                raise TaichiRuntimeError(
                    "A cold TaskLaunchPolicy specialization must be prepared "
                    "on the Python main thread; call bound.report(*args) once "
                    "before concurrent launches"
                )
            self.materialize(
                key=key,
                args=args,
                arg_features=arg_features,
                task_launch_policy=policy,
            )
            return key

    def _validate_task_launch_policy_specialization(self, key, policy):
        """Compile and validate a cold policy specialization without enqueueing it."""
        from taichi_forge.lang.task_manifest import OffloadedTaskManifest

        cached = self._task_launch_policy_manifests.get(key)
        if cached is not None:
            return cached
        if threading.current_thread() is not threading.main_thread():
            raise TaichiRuntimeError(
                "A cold TaskLaunchPolicy specialization must be prepared "
                "on the Python main thread; call bound.report(*args) once "
                "before concurrent launches"
            )

        kernel_cpp = self.compiled_kernels[key]
        raw = self.runtime.prog._kernel_task_manifest(kernel_cpp)
        tasks = tuple(OffloadedTaskManifest._from_core(item) for item in raw)
        range_tasks = tuple(task for task in tasks if task.task_type == "range_for")
        if len(range_tasks) != 1:
            raise TaichiRuntimeError(
                "TaskLaunchPolicy specialization did not produce exactly one "
                "parallel range task"
            )
        selected = range_tasks[0].selected_block_size
        if selected is None:
            raise TaichiRuntimeError(
                "TaskLaunchPolicy backend did not expose a selected block size"
            )
        if policy.mode == "require" and selected != policy.block_dim:
            raise TaichiRuntimeError(
                f"TaskLaunchPolicy require(block_dim={policy.block_dim}) was not "
                f"satisfied; backend selected {selected}"
            )
        self._task_launch_policy_manifests[key] = tasks
        return tasks

    @staticmethod
    def _task_launch_backend_kind():
        backend = _ti_core.arch_name(impl.current_cfg().arch)
        if backend in ("cuda", "vulkan"):
            return backend, "native"
        if backend in ("x64", "arm64"):
            return backend, "cpu"
        return backend, "unsupported"

    def with_launch_policy(self, policy):
        """Bind an immutable TaskLaunchPolicy without changing normal calls."""

        from taichi_forge.lang.task_launch import TaskLaunchPolicy

        if not isinstance(policy, TaskLaunchPolicy):
            raise TypeError("with_launch_policy expects a TaskLaunchPolicy")
        return _TaskLaunchBinding(self, policy)

    def _call_with_task_launch_policy(self, policy, *args, **kwargs):
        backend, kind = self._task_launch_backend_kind()
        if policy.mode == "auto" or (kind == "cpu" and policy.mode == "hint"):
            return self(*args, **kwargs)
        if kind == "cpu":
            raise TaichiRuntimeError(
                "TaskLaunchPolicy require is unavailable on CPU: the CPU runtime "
                "uses a worker scheduler rather than a GPU block"
            )
        if kind != "native":
            raise TaichiRuntimeError(
                f"TaskLaunchPolicy is unavailable on backend {backend}"
            )
        if self.autodiff_mode != AutodiffMode.NONE:
            raise TaichiRuntimeError(
                "TaskLaunchPolicy supports primal direct JIT kernels only"
            )
        if (
            self.runtime.target_tape is not None
            or self.runtime.fwd_mode_manager is not None
            or self.runtime.grad_replaced
        ):
            raise TaichiRuntimeError(
                "TaskLaunchPolicy cannot be used inside an automatic "
                "differentiation context"
            )

        args = _process_args(self, args, kwargs)
        key = self._ensure_compiled_with_task_launch_policy(policy, *args)
        self._validate_task_launch_policy_specialization(key, policy)
        kernel_cpp = self.compiled_kernels[key]
        return self.launch_kernel(kernel_cpp, *args)

    def _task_launch_report(self, policy, *args, **kwargs):
        from taichi_forge.lang.task_launch import (
            TaskLaunchReport,
            _task_launch_resource_reports,
        )

        backend, kind = self._task_launch_backend_kind()
        if policy.mode == "auto":
            tasks = self.task_manifest(*args, **kwargs)
            return TaskLaunchReport(
                policy=policy,
                backend=backend,
                status="auto",
                reason="compiler/backend default geometry",
                tasks=tasks,
                resources=_task_launch_resource_reports(
                    tasks, policy, "auto", impl.current_cfg()
                ),
            )
        if kind == "cpu":
            if policy.mode == "require":
                raise TaichiRuntimeError(
                    "TaskLaunchPolicy require is unavailable on CPU: the CPU "
                    "runtime uses a worker scheduler rather than a GPU block"
                )
            tasks = self.task_manifest(*args, **kwargs)
            return TaskLaunchReport(
                policy=policy,
                backend=backend,
                status="fallback_auto",
                reason="CPU has no GPU block geometry; hint preserved auto scheduling",
                tasks=tasks,
                resources=_task_launch_resource_reports(
                    tasks, policy, "fallback_auto", impl.current_cfg()
                ),
            )
        if kind != "native":
            raise TaichiRuntimeError(
                f"TaskLaunchPolicy is unavailable on backend {backend}"
            )

        processed = _process_args(self, args, kwargs)
        key = self._ensure_compiled_with_task_launch_policy(policy, *processed)
        tasks = self._validate_task_launch_policy_specialization(key, policy)
        range_tasks = tuple(task for task in tasks if task.task_type == "range_for")
        selected = range_tasks[0].selected_block_size
        applied = selected == policy.block_dim
        status = "applied" if applied else "hint_not_applied"
        return TaskLaunchReport(
            policy=policy,
            backend=backend,
            status=status,
            reason=(
                "backend selected the requested block size"
                if applied
                else "an explicit source-level loop_config or backend constraint won"
            ),
            tasks=tasks,
            resources=_task_launch_resource_reports(
                tasks, policy, status, impl.current_cfg()
            ),
        )

    def task_manifest(self, *args, **kwargs):
        """Return immutable metadata for this argument specialization.

        The query may compile a cold specialization, but never launches it or
        allocates device-side observation storage.
        """
        from taichi_forge.lang.task_manifest import OffloadedTaskManifest

        args = _process_args(self, args, kwargs)
        key = self.ensure_compiled(*args)
        kernel_cpp = self.compiled_kernels[key]
        raw = self.runtime.prog._kernel_task_manifest(kernel_cpp)
        return tuple(OffloadedTaskManifest._from_core(item) for item in raw)

    # For small kernels (< 3us), the performance can be pretty sensitive to overhead in __call__
    # Thus this part needs to be fast. (i.e. < 3us on a 4 GHz x64 CPU)
    @_shell_pop_print
    def __call__(self, *args, **kwargs):
        args = _process_args(self, args, kwargs)

        # A reverse kernel is already the result of one AD transform.  Running
        # it while an automatic AD context is active would request an
        # unverified higher-order transform (FwdMode) or mutate gradients that
        # the enclosing Tape owns.  Reject both cases before compilation or
        # device submission instead of relying on an assertion in FwdMode or
        # silently producing an incomplete derivative in Tape.
        if self.autodiff_mode == AutodiffMode.REVERSE:
            if self.runtime.fwd_mode_manager is not None:
                raise TaichiRuntimeError(
                    "Forward-on-reverse automatic differentiation is not "
                    "supported; call kernel.grad() outside ti.ad.FwdMode()."
                )
            if self.runtime.target_tape is not None:
                raise TaichiRuntimeError(
                    "Manual reverse kernel execution inside ti.ad.Tape() is "
                    "not supported; let the Tape run recorded adjoints when "
                    "the context exits."
                )

        # Transform the primal kernel to forward mode grad kernel
        # then recover to primal when exiting the forward mode manager
        if self.runtime.fwd_mode_manager and not self.runtime.grad_replaced:
            self.runtime.fwd_mode_manager.insert(self)

        # Both the class kernels and the plain-function kernels are unified now.
        # In both cases, |self.grad| is another Kernel instance that computes the
        # gradient. For class kernels, args[0] is always the kernel owner.

        # No need to capture grad kernels because they are already bound with their primal kernels
        if (
            self.autodiff_mode in (AutodiffMode.NONE, AutodiffMode.VALIDATION)
            and self.runtime.target_tape
            and not self.runtime.grad_replaced
        ):
            self.runtime.target_tape.insert(self, args)

        if self.autodiff_mode != AutodiffMode.NONE and impl.current_cfg().opt_level == 0:
            _logging.warn("""opt_level = 1 is enforced to enable gradient computation.""")
            impl.current_cfg().opt_level = 1
        key = self.ensure_compiled(*args)
        kernel_cpp = self.compiled_kernels[key]
        allocate_all_external_grad = (
            self.autodiff_mode != AutodiffMode.NONE
            or self.runtime.target_tape is not None
            or self.runtime.fwd_mode_manager is not None
        )
        return self.launch_kernel(
            kernel_cpp,
            *args,
            _allocate_all_external_grad=allocate_all_external_grad,
            _explicit_external_grad_args=self._external_grad_accesses.get(
                key, frozenset()
            ),
        )


class _TaskLaunchBinding:
    """A reusable policy-bound view of a direct JIT kernel."""

    def __init__(self, kernel, policy, bound_args=()):
        self._kernel = kernel
        self.policy = policy
        self._bound_args = tuple(bound_args)
        self._fast_runtime = None
        self._fast_key = None
        self._fast_kernel_cpp = None
        self._fallback_auto = policy.mode == "auto"
        self.__name__ = kernel.func.__name__

    def _refresh_fast_path(self, report=None):
        if report is not None and report.status == "fallback_auto":
            self._fallback_auto = True
            self._fast_runtime = self._kernel.runtime
            return
        if self.policy.mode == "auto":
            return
        self._fast_runtime = self._kernel.runtime
        if self._kernel.mapper._dynamic_arg_extractors:
            return
        key = (
            self._kernel.func,
            0,
            self._kernel.autodiff_mode,
            self.policy._specialization_key,
        )
        if key in self._kernel._task_launch_policy_manifests:
            self._fast_runtime = self._kernel.runtime
            self._fast_key = key
            self._fast_kernel_cpp = self._kernel.compiled_kernels[key]

    def __call__(self, *args, **kwargs):
        try:
            runtime = self._kernel.runtime
            if self.policy.mode == "auto" or (
                self._fallback_auto and self._fast_runtime is runtime
            ):
                return self._kernel(*self._bound_args, *args, **kwargs)
            if (
                self._fast_runtime is runtime
                and runtime.target_tape is None
                and runtime.fwd_mode_manager is None
                and not runtime.grad_replaced
            ):
                processed = _process_args(
                    self._kernel, (*self._bound_args, *args), kwargs
                )
                if (
                    self._fast_kernel_cpp is not None
                    and self._kernel.compiled_kernels.get(self._fast_key)
                    is self._fast_kernel_cpp
                ):
                    return self._kernel.launch_kernel(
                        self._fast_kernel_cpp, *processed
                    )
                key = self._kernel._ensure_compiled_with_task_launch_policy(
                    self.policy, *processed
                )
                self._kernel._validate_task_launch_policy_specialization(
                    key, self.policy
                )
                return self._kernel.launch_kernel(
                    self._kernel.compiled_kernels[key], *processed
                )
            result = self._kernel._call_with_task_launch_policy(
                self.policy, *self._bound_args, *args, **kwargs
            )
            if (
                self.policy.mode == "hint"
                and self._kernel._task_launch_backend_kind()[1] == "cpu"
            ):
                self._fallback_auto = True
                self._fast_runtime = runtime
            else:
                self._refresh_fast_path()
            return result
        except (TaichiCompilationError, TaichiRuntimeError) as exc:
            if impl.get_runtime().print_full_traceback:
                raise
            raise type(exc)("\n" + str(exc)) from None

    def report(self, *args, **kwargs):
        """Compile if needed and report resolution without submitting work."""

        report = self._kernel._task_launch_report(
            self.policy, *self._bound_args, *args, **kwargs
        )
        self._refresh_fast_path(report)
        return report

    def task_manifest(self, *args, **kwargs):
        """Return the policy specialization's physical task manifest."""

        return self.report(*args, **kwargs).tasks


# For a Taichi class definition like below:
#
# @ti.data_oriented
# class X:
#   @ti.kernel
#   def foo(self):
#     ...
#
# When ti.kernel runs, the stackframe's |code_context| of Python 3.8(+) is
# different from that of Python 3.7 and below. In 3.8+, it is 'class X:',
# whereas in <=3.7, it is '@ti.data_oriented'. More interestingly, if the class
# inherits, i.e. class X(object):, then in both versions, |code_context| is
# 'class X(object):'...
_KERNEL_CLASS_STACKFRAME_STMT_RES = [
    re.compile(r"@(\w+\.)?data_oriented"),
    re.compile(r"class "),
]


def _inside_class(level_of_class_stackframe):
    try:
        maybe_class_frame = sys._getframe(level_of_class_stackframe)
        statement_list = inspect.getframeinfo(maybe_class_frame)[3]
        first_statment = statement_list[0].strip()
        for pat in _KERNEL_CLASS_STACKFRAME_STMT_RES:
            if pat.match(first_statment):
                return True
    except:
        pass
    return False


def _kernel_impl(_func, level_of_class_stackframe, verbose=False, opt_level=None):
    # Can decorators determine if a function is being defined inside a class?
    # https://stackoverflow.com/a/8793684/12003165
    is_classkernel = _inside_class(level_of_class_stackframe + 1)

    if verbose:
        print(f"kernel={_func.__name__} is_classkernel={is_classkernel}")
    primal = Kernel(_func, autodiff_mode=AutodiffMode.NONE, _classkernel=is_classkernel, opt_level=opt_level)
    adjoint = Kernel(_func, autodiff_mode=AutodiffMode.REVERSE, _classkernel=is_classkernel, opt_level=opt_level)
    # Having |primal| contains |grad| makes the tape work.
    primal.grad = adjoint

    if is_classkernel:
        # For class kernels, their primal/adjoint callables are constructed
        # when the kernel is accessed via the instance inside
        # _BoundedDifferentiableMethod.
        # This is because we need to bind the kernel or |grad| to the instance
        # owning the kernel, which is not known until the kernel is accessed.
        #
        # See also: _BoundedDifferentiableMethod, data_oriented.
        @functools.wraps(_func)
        def wrapped(*args, **kwargs):
            # If we reach here (we should never), it means the class is not decorated
            # with @ti.data_oriented, otherwise getattr would have intercepted the call.
            clsobj = type(args[0])
            assert not hasattr(clsobj, "_data_oriented")
            raise TaichiSyntaxError(f"Please decorate class {clsobj.__name__} with @ti.data_oriented")

    else:

        @functools.wraps(_func)
        def wrapped(*args, **kwargs):
            try:
                return primal(*args, **kwargs)
            except (TaichiCompilationError, TaichiRuntimeError) as e:
                if impl.get_runtime().print_full_traceback:
                    raise e
                raise type(e)("\n" + str(e)) from None

        wrapped.grad = adjoint

    wrapped._is_wrapped_kernel = True
    wrapped._is_classkernel = is_classkernel
    wrapped._primal = primal
    wrapped._adjoint = adjoint
    if not is_classkernel:
        wrapped.task_manifest = primal.task_manifest
        wrapped.with_launch_policy = primal.with_launch_policy
    return wrapped


def kernel(fn=None, *, opt_level=None):
    """Marks a function as a Taichi kernel.

    A Taichi kernel is a function written in Python, and gets JIT compiled by
    Taichi into native CPU/GPU instructions (e.g. a series of CUDA kernels).
    The top-level ``for`` loops are automatically parallelized, and distributed
    to either a CPU thread pool or massively parallel GPUs.

    Kernel's gradient kernel would be generated automatically by the AutoDiff system.

    See also https://docs.taichi-lang.org/docs/syntax#kernel.

    Args:
        fn (Callable): the Python function to be decorated
        opt_level (Optional[str]): per-kernel ``compile_tier`` override.
            One of ``"fast"`` / ``"balanced"`` / ``"full"`` or ``None`` (default,
            inherits ``ti.init(compile_tier=...)``). ``"fast"`` skips expensive
            IR + LLVM passes — recommended for cold-path / serial / I/O-bound
            kernels where compile time dominates and runtime is insensitive.

    Returns:
        Callable: The decorated function

    Example::

        >>> x = ti.field(ti.i32, shape=(4, 8))
        >>>
        >>> @ti.kernel
        >>> def run():
        >>>     for i in x:
        >>>         x[i] = i
        >>>
        >>> @ti.kernel(opt_level="fast")
        >>> def cold_path_kernel():
        >>>     # Compiled with reduced optimization for faster compile time.
        >>>     pass
    """
    # Support both bare `@ti.kernel` and parameterized `@ti.kernel(opt_level=...)`.
    # When called bare, fn is the user function (level_of_class_stackframe=3:
    # _kernel_impl -> kernel -> user). When called with parens, fn is None and
    # we return the inner decorator (level_of_class_stackframe=4:
    # _kernel_impl -> _decorator -> kernel(...) call site -> user).
    if fn is not None:
        return _kernel_impl(fn, level_of_class_stackframe=3, opt_level=opt_level)

    def _decorator(_fn):
        return _kernel_impl(_fn, level_of_class_stackframe=4, opt_level=opt_level)

    return _decorator


class _BoundedDifferentiableMethod:
    def __init__(self, kernel_owner, wrapped_kernel_func):
        clsobj = type(kernel_owner)
        if not getattr(clsobj, "_data_oriented", False):
            raise TaichiSyntaxError(f"Please decorate class {clsobj.__name__} with @ti.data_oriented")
        self._kernel_owner = kernel_owner
        self._primal = wrapped_kernel_func._primal
        self._adjoint = wrapped_kernel_func._adjoint
        self._is_staticmethod = wrapped_kernel_func._is_staticmethod
        self.__name__ = None

    def __call__(self, *args, **kwargs):
        try:
            if self._is_staticmethod:
                return self._primal(*args, **kwargs)
            return self._primal(self._kernel_owner, *args, **kwargs)
        except (TaichiCompilationError, TaichiRuntimeError) as e:
            if impl.get_runtime().print_full_traceback:
                raise e
            raise type(e)("\n" + str(e)) from None

    def grad(self, *args, **kwargs):
        return self._adjoint(self._kernel_owner, *args, **kwargs)

    def task_manifest(self, *args, **kwargs):
        if self._is_staticmethod:
            return self._primal.task_manifest(*args, **kwargs)
        return self._primal.task_manifest(
            self._kernel_owner, *args, **kwargs
        )

    def with_launch_policy(self, policy):
        from taichi_forge.lang.task_launch import TaskLaunchPolicy

        if not isinstance(policy, TaskLaunchPolicy):
            raise TypeError("with_launch_policy expects a TaskLaunchPolicy")
        bound_args = () if self._is_staticmethod else (self._kernel_owner,)
        return _TaskLaunchBinding(self._primal, policy, bound_args)


def data_oriented(cls):
    """Marks a class as Taichi compatible.

    To allow for modularized code, Taichi provides this decorator so that
    Taichi kernels can be defined inside a class.

    See also https://docs.taichi-lang.org/docs/odop

    Example::

        >>> @ti.data_oriented
        >>> class TiArray:
        >>>     def __init__(self, n):
        >>>         self.x = ti.field(ti.f32, shape=n)
        >>>
        >>>     @ti.kernel
        >>>     def inc(self):
        >>>         for i in self.x:
        >>>             self.x[i] += 1.0
        >>>
        >>> a = TiArray(32)
        >>> a.inc()

    Args:
        cls (Class): the class to be decorated

    Returns:
        The decorated class.
    """

    def _getattr(self, item):
        method = cls.__dict__.get(item, None)
        is_property = method.__class__ == property
        is_staticmethod = method.__class__ == staticmethod
        if is_property:
            x = method.fget
        else:
            x = super(cls, self).__getattribute__(item)
        if hasattr(x, "_is_wrapped_kernel"):
            if inspect.ismethod(x):
                wrapped = x.__func__
            else:
                wrapped = x
            wrapped._is_staticmethod = is_staticmethod
            assert inspect.isfunction(wrapped)
            if wrapped._is_classkernel:
                ret = _BoundedDifferentiableMethod(self, wrapped)
                ret.__name__ = wrapped.__name__
                if is_property:
                    return ret()
                return ret
        if is_property:
            return x(self)
        return x

    cls.__getattribute__ = _getattr
    cls._data_oriented = True

    return cls


__all__ = ["data_oriented", "func", "kernel", "pyfunc", "real_func"]
