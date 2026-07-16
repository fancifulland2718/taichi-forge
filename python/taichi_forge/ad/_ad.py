"""Taichi automatic differentiation module.

This module supplies two decorators for users to customize their
gradient computation task.
"""

import warnings
from math import prod

import numpy as np
import taichi_forge.types.primitive_types as types
from taichi_forge.lang import impl
from taichi_forge.lang.enums import AutodiffMode, SNodeGradType
from taichi_forge.lang.expr import Expr
from taichi_forge.lang.field import Field, ScalarField
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang._ndarray import Ndarray
from taichi_forge.lang.kernel_impl import kernel
from taichi_forge.lang.snode import SNode
from taichi_forge.lang.util import to_numpy_type
from taichi_forge.types import ndarray, template

from taichi_forge import _snode


class GradChecker:
    def __init__(self, loss, to_check):
        self.to_check = to_check
        self.loss = loss
        self.eps_range = 2.0 ** np.arange(-3, -30, -2).astype(np.float64)
        self.result = [None] * len(to_check)
        self.all_fields = get_all_fields()
        self.backups = save_all_fields(self.all_fields)

    def add_calls(self, calls):
        self.calls = calls

    def check_grad(self):
        assert self.loss.dtype == types.f64, "Only f64 is supported when checking grad."

        @kernel
        def x_pos(x: template(), tangent_np: ndarray(), eps: types.f64):
            for I in impl.grouped(x):
                x[I] += eps * tangent_np[I]

        @kernel
        def x_neg(x: template(), tangent_np: ndarray(), eps: types.f64):
            for I in impl.grouped(x):
                x[I] -= eps * tangent_np[I]

        for i, x in enumerate(self.to_check):
            if x is self.loss:
                self.result[i] = True
                continue

            check_pass = False

            re_range = []
            for eps in self.eps_range:
                tangent_np = np.array(np.random.rand(*x.shape)).astype(np.float64)

                restore_all_fields(self.all_fields, self.backups)
                x_pos(x, tangent_np, eps)
                for func, args in self.calls:
                    func(*args)
                loss_pos = self.loss.to_numpy()

                restore_all_fields(self.all_fields, self.backups)
                x_neg(x, tangent_np, eps)
                for func, args in self.calls:
                    func(*args)
                loss_neg = self.loss.to_numpy()

                ip_numerical = (loss_pos - loss_neg) * 0.5 / eps
                x_grad_np = x.grad.to_numpy()
                extra_dim = x_grad_np.ndim - tangent_np.ndim
                if extra_dim == 1:
                    tangent_np = np.expand_dims(tangent_np, axis=-1)
                if extra_dim == 2:
                    tangent_np = np.expand_dims(tangent_np, axis=-1)

                ip_autodiff = np.sum(x_grad_np * tangent_np)
                err = abs(ip_autodiff - ip_numerical)
                if ip_numerical != 0:
                    re = err / abs(ip_autodiff)
                else:
                    re = err / (abs(ip_autodiff) + 1e-20)
                re_range.append(re)

                if err * 100 <= abs(ip_autodiff):
                    check_pass = True
                    break

            self.result[i] = check_pass

            if not check_pass:
                print(
                    "variable",
                    i,
                    "has relative error",
                    min(re_range),
                    ", expected relative error 0.01",
                )
            else:
                print("variable", i, "passes grad check")

        assert all(self.result), "Grad check failed: Not all variables pass grad check"

        restore_all_fields(self.all_fields, self.backups)
        for func, args in self.calls:
            func(*args)


def get_all_fields():
    def visit(node, fields):
        for _i in range(node.ptr.get_num_ch()):
            ch = node.ptr.get_ch(_i)
            if not ch.is_place():
                visit(SNode(ch), fields)
            else:
                if not ch.is_primal():
                    continue
                fields.append(ScalarField(Expr(ch.get_expr())))

    fields = []
    for root_fb in _snode.FieldsBuilder._finalized_roots():
        visit(root_fb, fields)
    return fields


def save_all_fields(all_fields):
    return [x.to_numpy() for x in all_fields]


def restore_all_fields(all_fields, backups):
    assert len(all_fields) == len(backups)
    for f, x in zip(all_fields, backups):
        f.from_numpy(x)


class Tape:
    def __init__(self, loss=None, clear_gradients=True, validation=False, grad_check=None):
        """A context manager for reverse mode autodiff :class:`~taichi_forge.ad.Tape`. The
        context manager would catching all of the callings of functions that
        decorated by :func:`~taichi_forge.lang.kernel_impl.kernel` or
        :func:`~taichi_forge.ad.grad_replaced` under `with` statement, and calculate
        all the partial gradients of a given loss variable by calling all of the
        gradient function of the callings caught in reverse order while `with`
        statement ended.

        See also :func:`~taichi_forge.lang.kernel_impl.kernel` and
        :func:`~taichi_forge.ad.grad_replaced` for gradient functions.

        Args:
            loss(:class:`~taichi_forge.lang.expr.Expr`): The loss field, which shape should be ().
            clear_gradients(Bool): Before `with` body start, clear all gradients or not.
            validation(Bool): Check whether the code inside the context manager is autodiff valid, e.g., agree with the global data access rule.
            grad_check(List[Field]): List of fields that need to check gradients.

        Example::

            >>> @ti.kernel
            >>> def sum(a: ti.float32):
            >>>     for I in ti.grouped(x):
            >>>         y[None] += x[I] ** a
            >>>
            >>> with ti.ad.Tape(loss = y):
            >>>     sum(2)
        """
        self.calls = []
        self.modes = []
        self.entered = False
        self.gradient_evaluated = False
        self.clear_gradients = clear_gradients
        self.validation = validation
        self.runtime = impl.get_runtime()
        if not self.runtime.prog.config().debug and self.validation:
            warnings.warn(
                "Debug mode is disabled, autodiff valid check will not work. Please specify `ti.init(debug=True)` to enable the check.",
                Warning,
            )
        self.eval_on_exit = loss is not None
        self.loss = loss
        self.grad_checker = None
        if grad_check:
            assert isinstance(grad_check, list), "grad_check should be a list of fields that need to check gradients."
            self.grad_checker = GradChecker(loss, grad_check)

    def __enter__(self):
        if self.entered:
            raise TaichiRuntimeError("A ti.ad.Tape() instance can be entered only once.")
        submission_state = self.runtime._active_graph_submissions
        if submission_state > 0:
            raise TaichiRuntimeError(
                "Cannot enter ti.ad.Tape() while another Python thread has an "
                "active Graph.run() host submission. Wait for that submission "
                "to return before starting automatic differentiation."
            )
        if (
            submission_state < 0
            or self.runtime.target_tape is not None
            or self.runtime.fwd_mode_manager is not None
        ):
            raise TaichiRuntimeError(
                "Cannot enter ti.ad.Tape() while another automatic AD context "
                "is active or being initialized. Runtime AD state is global."
            )
        self.runtime._active_graph_submissions = -1
        try:
            return self._enter_impl()
        finally:
            self.runtime._active_graph_submissions = 0

    def _enter_impl(self):
        self.entered = True

        if isinstance(self.loss, Field):
            impl.get_runtime().materialize()
            if len(self.loss.shape) != 0:
                raise RuntimeError("The loss of `Tape` must be a 0-D field, i.e. scalar")
            if not self.loss.snode.ptr.has_adjoint():
                raise RuntimeError(
                    "Gradients of loss are not allocated, please use ti.field(..., needs_grad=True)"
                    " for all fields that are required by autodiff."
                )
            if self.clear_gradients:
                clear_all_gradients()
            if self.validation:
                clear_all_gradients(gradient_type=SNodeGradType.ADJOINT_CHECKBIT)

            self.loss.fill(0.0)
        elif isinstance(self.loss, Ndarray):
            if self.loss._get_nelement() != 1:
                raise RuntimeError("The loss of `Tape` must be an ndarray with only one element")
            if self.loss.grad is None:
                raise RuntimeError(
                    "Gradients of loss are not allocated, please set needs_grad=True for all ndarrays that are required by autodiff."
                )
            self.loss.fill(0.0)
        else:
            import torch  # pylint: disable=C0415

            if self.loss.numel() != 1:
                raise RuntimeError("The loss of `Tape` must be a tensor only contains one element")
            if not self.loss.requires_grad:
                raise RuntimeError(
                    "Gradients of loss are not allocated, please set requires_grad=True for all tensors that are required by autodiff."
                )
            with torch.no_grad():
                self.loss.fill_(0.0)

        # Attach the context manager to runtime
        self.runtime.target_tape = self
        return self

    def __exit__(self, _type, value, tb):
        self.runtime.target_tape = None
        try:
            # Do not launch adjoints after the context body failed.  Besides
            # masking the original exception, those kernels would consume a
            # potentially partial primal trace.
            if self.eval_on_exit and _type is None:
                self.grad()
        finally:
            for calls, mode in zip(self.calls, self.modes):
                if mode is not None:
                    calls[0].autodiff_mode = mode

    def insert(self, func, args):
        # Kernels with mode `AutodiffMode.NONE` and `AutodiffMode.VALIDATION` are all forward kernels.
        # The difference is there are `assert` for global data access rule check in VALIDATION kernels.
        if func.autodiff_mode not in (
            AutodiffMode.NONE,
            AutodiffMode.VALIDATION,
        ):
            raise TaichiRuntimeError(
                "ti.ad.Tape() can record only primal kernels; higher-order "
                "automatic differentiation is not supported."
            )
        self.modes.append(func.autodiff_mode)
        if self.validation:
            func.autodiff_mode = AutodiffMode.VALIDATION
        self.calls.append((func, args))

    def insert_native(self, record):
        self.modes.append(None)
        self.calls.append((record, ()))

    def grad(self):
        assert self.entered, "Before evaluating gradients tape must be entered."
        assert not self.gradient_evaluated, "Gradients of grad can be evaluated only once."

        # Set grad for loss
        if isinstance(self.loss, (Field, Ndarray)):
            self.loss.grad.fill(1.0)
            from taichi_forge.lang.misc import vulkan  # pylint: disable=C0415

            if impl.current_cfg().arch == vulkan:
                from taichi_forge.lang.runtime_ops import sync  # pylint: disable=C0415

                sync()
        else:
            import torch  # pylint: disable=C0415

            if self.loss.grad is None:
                self.loss.grad = torch.ones_like(self.loss)
            else:
                with torch.no_grad():
                    self.loss.grad.fill_(1.0)

        for func, args in reversed(self.calls):
            # we need to check whether "func" has "grad" attribute
            # since we insert write_int and write_float kernels to self.calls
            # e.g. x[None] = 0.0, this func has no grad attribute
            if hasattr(func, "grad"):
                func.grad(*args)

        self.gradient_evaluated = True
        if self.grad_checker:
            self.grad_checker.add_calls(self.calls)
            self.grad_checker.check_grad()


def clear_all_gradients(gradient_type=SNodeGradType.ADJOINT):
    """Sets the gradients of all fields to zero."""
    impl.get_runtime().materialize()

    from taichi_forge.lang.field import (  # pylint: disable=C0415
        _dense_host_copy_value_type,
        _dense_native_field_is_contiguous,
        _dense_native_method_descriptor,
        _dense_value_type_size,
    )
    from taichi_forge.lang.misc import vulkan  # pylint: disable=C0415

    arch = impl.current_cfg().arch
    vulkan_batch_method_name = "vulkan_zero_dense_fields" if arch == vulkan else None
    used_vulkan_native_clear = False

    def try_dense_native_clear_field(field):
        nonlocal used_vulkan_native_clear
        try:
            cleared = field._try_native_dense_fill(0)
        except RuntimeError:
            return False
        if cleared and arch == vulkan:
            used_vulkan_native_clear = True
        return cleared

    def try_dense_native_clear_expr(expr):
        return try_dense_native_clear_field(ScalarField(Expr(expr)))

    def dense_packed_clear_info(node):
        num_children = node.ptr.get_num_ch()
        if num_children <= 1:
            return None
        try:
            node_key = (int(node.ptr.get_snode_tree_id()), int(node.ptr.id))
        except (AttributeError, TypeError, ValueError):
            return None
        fields = []
        for child_index in range(num_children):
            ch = node.ptr.get_ch(child_index)
            if not ch.is_place() or ch.get_snode_grad_type() != gradient_type:
                return None
            field = ScalarField(Expr(ch.get_expr()))
            fields.append(field)
        value_type = _dense_host_copy_value_type(fields[0].dtype)
        value_size = _dense_value_type_size(value_type)
        if value_type is None or value_size == 0:
            return None
        shape = tuple(fields[0].shape)
        if len(shape) == 0:
            return None
        parent = fields[0].snode.parent()
        parent_key = (
            None
            if parent is None
            else (int(parent._snode_tree_id), int(parent._id))
        )
        if (
            parent is None
            or parent_key != node_key
            or fields[0].snode.parent(2) is not impl.root
        ):
            return None
        lane_count = len(fields)
        expected_stride = lane_count * value_size
        if (
            parent._cell_size_bytes != expected_stride
            or fields[0].snode._offset_bytes_in_parent_cell != 0
        ):
            return None
        for lane, field in enumerate(fields):
            snode = field.snode
            snode_parent = snode.parent()
            snode_parent_key = (
                None
                if snode_parent is None
                else (int(snode_parent._snode_tree_id), int(snode_parent._id))
            )
            if (
                field.dtype != fields[0].dtype
                or tuple(field.shape) != shape
                or snode_parent_key != node_key
                or snode.parent(2) is not impl.root
                or snode._offset_bytes_in_parent_cell != lane * value_size
            ):
                return None
        n = int(np.prod(shape, dtype=np.int64))
        return fields[0], value_type, n, lane_count

    def try_dense_native_clear_packed_node(node):
        nonlocal used_vulkan_native_clear
        clear_info = dense_packed_clear_info(node)
        if clear_info is None:
            return False
        field, value_type, n, lane_count = clear_info
        prog = impl.get_runtime().prog
        method = _dense_native_method_descriptor(prog, "fill_dense_field_packed")
        if method is None:
            return False
        try:
            method(prog, field.snode.ptr, value_type, 0, n, lane_count)
        except RuntimeError:
            return False
        if arch == vulkan:
            used_vulkan_native_clear = True
        return True

    def dense_clear_info(ch):
        field = ScalarField(Expr(ch.get_expr()))
        value_type = _dense_host_copy_value_type(field.dtype)
        if value_type is None:
            return None
        n = int(np.prod(field.shape, dtype=np.int64))
        return field, value_type, n

    def make_vulkan_native_clear_job(ch):
        if vulkan_batch_method_name is None:
            return None
        clear_info = dense_clear_info(ch)
        if clear_info is None:
            return None
        field, value_type, n = clear_info
        if not _dense_native_field_is_contiguous(field, value_type):
            return None
        return field, field.snode.ptr, value_type, n, ch.get_expr()

    def try_vulkan_native_clear_batch(jobs):
        nonlocal used_vulkan_native_clear
        if not jobs:
            return True
        method = getattr(impl.get_runtime().prog, vulkan_batch_method_name, None)
        if method is None:
            return False
        try:
            method(
                [job[1] for job in jobs],
                [job[2] for job in jobs],
                [job[3] for job in jobs],
            )
        except RuntimeError:
            return False
        used_vulkan_native_clear = True
        return True

    def try_vulkan_native_clear_jobs(jobs):
        if not jobs:
            return []
        if try_vulkan_native_clear_batch(jobs):
            return []
        fallback_places = []
        for field, _, _, _, expr in jobs:
            if not try_dense_native_clear_field(field):
                fallback_places.append(expr)
        return fallback_places

    def flush_vulkan_jobs(jobs, places):
        if jobs:
            places.extend(try_vulkan_native_clear_jobs(jobs))
            jobs.clear()

    def visit(node):
        if try_dense_native_clear_packed_node(node):
            return
        places = []
        vulkan_jobs = []
        for _i in range(node.ptr.get_num_ch()):
            ch = node.ptr.get_ch(_i)
            if not ch.is_place():
                flush_vulkan_jobs(vulkan_jobs, places)
                visit(SNode(ch))
            else:
                if ch.get_snode_grad_type() == gradient_type:
                    vulkan_job = make_vulkan_native_clear_job(ch)
                    if vulkan_job is not None:
                        vulkan_jobs.append(vulkan_job)
                    elif not try_dense_native_clear_expr(ch.get_expr()):
                        places.append(ch.get_expr())

        flush_vulkan_jobs(vulkan_jobs, places)

        places = tuple(places)
        if places:
            from taichi_forge._kernels import clear_gradients  # pylint: disable=C0415

            clear_gradients(places)

    for root_fb in _snode.FieldsBuilder._finalized_roots():
        visit(root_fb)

    if used_vulkan_native_clear:
        from taichi_forge.lang.runtime_ops import sync  # pylint: disable=C0415

        sync()


def grad_replaced(func):
    """A decorator for python function to customize gradient with Taichi's autodiff
    system, e.g. `ti.ad.Tape()` and `kernel.grad()`.

    This decorator forces Taichi's autodiff system to use a user-defined gradient
    function for the decorated function. Its customized gradient must be decorated
    by :func:`~taichi_forge.ad.grad_for`.

    Args:
        fn (Callable): The python function to be decorated.

    Returns:
        Callable: The decorated function.

    Example::

        >>> @ti.kernel
        >>> def multiply(a: ti.float32):
        >>>     for I in ti.grouped(x):
        >>>         y[I] = x[I] * a
        >>>
        >>> @ti.kernel
        >>> def multiply_grad(a: ti.float32):
        >>>     for I in ti.grouped(x):
        >>>         x.grad[I] = y.grad[I] / a
        >>>
        >>> @ti.ad.grad_replaced
        >>> def foo(a):
        >>>     multiply(a)
        >>>
        >>> @ti.ad.grad_for(foo)
        >>> def foo_grad(a):
        >>>     multiply_grad(a)"""

    def decorated(*args, **kwargs):
        # TODO [#3025]: get rid of circular imports and move this to the top.
        impl.get_runtime().grad_replaced = True
        if impl.get_runtime().target_tape:
            impl.get_runtime().target_tape.insert(decorated, args)
        try:
            func(*args, **kwargs)
        finally:
            impl.get_runtime().grad_replaced = False

    decorated.grad = None
    decorated.autodiff_mode = AutodiffMode.NONE
    return decorated


def grad_for(primal):
    """Generates a decorator to decorate `primal`'s customized gradient function.

    See :func:`~taichi_forge.lang.grad_replaced` for examples.

    Args:
        primal (Callable): The primal function, must be decorated by :func:`~taichi_forge.ad.grad_replaced`.

    Returns:
        Callable: The decorator used to decorate customized gradient function."""

    def decorator(func):
        def decorated(*args, **kwargs):
            func(*args, **kwargs)

        if not hasattr(primal, "grad"):
            raise RuntimeError(f"Primal function `{primal.__name__}` must be decorated by ti.ad.grad_replaced")
        if primal.grad is not None:
            raise RuntimeError(
                "Primal function must be a **python** function instead of a taichi kernel. Please wrap the taichi kernel in a @ti.ad.grad_replaced decorated python function instead."
            )
        primal.grad = decorated
        return decorated

    return decorator


def no_grad(func):
    """A decorator for python function to skip gradient calculation within Taichi's
    autodiff system, e.g. `ti.ad.Tape()` and `kernel.grad()`.
    This decorator forces Taichi's autodiff system to use an empty gradient function
    for the decorated function.

    Args:
        fn (Callable): The python function to be decorated.

    Returns:
        Callable: The decorated function.

    Example::

        >>> @ti.kernel
        >>> def multiply(a: ti.float32):
        >>>     for I in ti.grouped(x):
        >>>         y[I] = x[I] * a
        >>>
        >>> @ti.no_grad
        >>> def foo(a):
        >>>     multiply(a)"""

    def decorated(*args, **kwargs):
        impl.get_runtime().grad_replaced = True
        if impl.get_runtime().target_tape:
            impl.get_runtime().target_tape.insert(decorated, args)
        try:
            func(*args, **kwargs)
        finally:
            impl.get_runtime().grad_replaced = False

    def placeholder(*args, **kwargs):
        return

    decorated.grad = placeholder
    decorated.autodiff_mode = AutodiffMode.NONE
    return decorated


class FwdMode:
    def __init__(self, loss, param, seed=None, clear_gradients=True):
        self.calls = []
        self.modes = []
        self.entered = False
        self.kernels_recovered = False
        self.runtime = impl.get_runtime()
        self.loss = loss
        self.param = param
        self.seed = seed
        self.clear_gradients = clear_gradients

    def __enter__(self):
        if self.entered:
            raise TaichiRuntimeError(
                "A ti.ad.FwdMode() instance can be entered only once."
            )
        submission_state = self.runtime._active_graph_submissions
        if submission_state > 0:
            raise TaichiRuntimeError(
                "Cannot enter ti.ad.FwdMode() while another Python thread has "
                "an active Graph.run() host submission. Wait for that "
                "submission to return before starting automatic differentiation."
            )
        if (
            submission_state < 0
            or self.runtime.target_tape is not None
            or self.runtime.fwd_mode_manager is not None
        ):
            raise TaichiRuntimeError(
                "Cannot enter ti.ad.FwdMode() while another automatic AD "
                "context is active or being initialized. Runtime AD state is global."
            )
        self.runtime._active_graph_submissions = -1
        try:
            return self._enter_impl()
        finally:
            self.runtime._active_graph_submissions = 0

    def _enter_impl(self):
        self.entered = True
        impl.get_runtime().materialize()
        if not isinstance(self.loss, list):
            self.loss = [self.loss]
        for ls in self.loss:
            assert isinstance(ls, ScalarField)

        # Currently we support only one N-D field as a group of parameters,
        # which is sufficient for computing Jacobian-vector product(Jvp).
        # For cases with multiple groups of parameters, it requires to run the forward ad multiple times,
        # which is out of scope of the current design for this interface.

        from taichi_forge.lang.matrix import MatrixField  # pylint: disable=C0415

        if isinstance(self.param, ScalarField):
            element_shape = ()
        elif isinstance(self.param, MatrixField):
            element_shape = (
                (self.param.n,)
                if self.param.ndim == 1
                else (self.param.n, self.param.m)
            )
            member_dtypes = tuple(
                ScalarField(member).dtype for member in self.param.vars
            )
            if any(dtype != member_dtypes[0] for dtype in member_dtypes[1:]):
                raise TaichiRuntimeError(
                    "ti.ad.FwdMode() requires a homogeneous VectorField/MatrixField parameter dtype"
                )
        else:
            raise TaichiRuntimeError(
                "ti.ad.FwdMode() param must be a ScalarField, VectorField, or MatrixField"
            )
        if self.param.dual is None:
            raise TaichiRuntimeError(
                "ti.ad.FwdMode() parameter has no dual storage; create it with needs_dual=True or ti.root.lazy_dual()"
            )

        seed_shape = tuple(self.param.shape) + element_shape
        parameter_count = prod(seed_shape) if seed_shape else 1

        if self.seed is None:
            if parameter_count == 1:
                # Compute the derivative respect to the first variable by default
                self.seed = [1.0]
            else:
                raise RuntimeError(
                    "`seed` is not set for non 0-D field, please specify."
                    " `seed` may be a flat sequence or an array shaped like the parameter field plus its element shape."
                    " E.g. Given a loss `loss = ti.field(float, shape=3)`, parameter `x = ti.field(float, shape=3)`"
                    "      seed = [0, 0, 1] indicates compute derivative respect to the third element of `x`."
                    "      seed = [1, 1, 1] indicates compute the sum of derivatives respect to all three element of `x`, i.e., Jacobian-vector product(Jvp) for each element in `loss`"
                )

        seed_array = np.asarray(self.seed, dtype=to_numpy_type(self.param.dtype))
        if seed_array.shape != seed_shape:
            if seed_array.ndim != 1 or seed_array.size != parameter_count:
                raise RuntimeError(
                    "ti.ad.FwdMode() seed shape mismatch: expected "
                    f"{seed_shape} or a flat sequence of length {parameter_count}, "
                    f"got {seed_array.shape}"
                )
            seed_array = seed_array.reshape(seed_shape)
        self._seed_array = np.ascontiguousarray(seed_array)

        # Clear gradients
        if self.clear_gradients:
            clear_all_gradients(gradient_type=SNodeGradType.DUAL)

        # Host seed order is field indices followed by row-major element
        # indices. MatrixField.from_numpy abstracts over AoS and SoA storage.
        if seed_shape:
            self.param.dual.from_numpy(self._seed_array)
        else:
            self.param.dual[None] = self._seed_array.item()

        # Attach the context manager to the runtime
        self.runtime.fwd_mode_manager = self

    def __exit__(self, _type, value, tb):
        self.runtime.fwd_mode_manager = None
        self.clear_seed()
        self.recover_kernels()

    def insert(self, func):
        if func.autodiff_mode not in (
            AutodiffMode.NONE,
            AutodiffMode.FORWARD,
        ):
            raise TaichiRuntimeError(
                "ti.ad.FwdMode() can transform only primal kernels; "
                "forward-on-reverse automatic differentiation is not supported."
            )
        self.modes.append(func.autodiff_mode)
        func.autodiff_mode = AutodiffMode.FORWARD
        self.calls.append((func))

    def recover_kernels(self):
        assert self.entered, "Before recover the kernels, fwd mode manager must be entered."
        for calls, mode in zip(self.calls, self.modes):
            calls.autodiff_mode = mode
        self.kernels_recovered = True

    def clear_seed(self):
        self.param.dual.fill(0)


__all__ = [
    "FwdMode",
    "Tape",
    "clear_all_gradients",
    "grad_for",
    "grad_replaced",
    "no_grad",
]
