import datetime
import hashlib
import os
from contextlib import contextmanager
from glob import glob
from pathlib import Path, PurePosixPath
from shutil import rmtree
from tempfile import mkdtemp
from urllib.parse import quote
from zipfile import ZipFile

import numpy as np

from taichi_forge._lib import core as _ti_core
from taichi_forge.aot.utils import (
    produce_injected_args,
    produce_injected_args_from_template,
    reject_acceleration_structure_arguments,
)
from taichi_forge.lang import impl, kernel_impl
from taichi_forge.lang._ndarray import Ndarray, StructNdarray
from taichi_forge.lang.enums import Layout
from taichi_forge.lang.exception import TaichiCompilationError, TaichiRuntimeError
from taichi_forge.lang.field import ScalarField
from taichi_forge.lang.matrix import MatrixField
from taichi_forge.types.annotations import template
from taichi_forge.types.ndarray_type import NdarrayType
from taichi_forge.types.texture_type import RWTextureType, TextureType

import taichi_forge


class KernelTemplate:
    def __init__(self, kernel_fn, aot_module):
        self._kernel_fn = kernel_fn
        self._aot_module = aot_module
        self._instantiated_keys = set()

    @staticmethod
    def _key_atom(value):
        return quote(str(value), safe="-_.")

    @staticmethod
    def keygen(v, key_p, fields):
        if isinstance(v, (int, float, bool)):
            key_p += "=" + KernelTemplate._key_atom(v) + "__"
            return key_p
        for ky, val in fields:
            if val is v:
                key_p += "=" + KernelTemplate._key_atom(ky) + "__"
                return key_p
        raise RuntimeError(
            "Arg type must be of type int/float/boolean" f" or taichi field. Type {str(type(v))}" " is not supported"
        )

    @staticmethod
    def _ndarray_keygen(v, anno, arg_name):
        if isinstance(v, StructNdarray):
            raise TaichiCompilationError(
                f"AOT ndarray template argument {arg_name} does not support "
                "StructNdarray; use a scalar/vector/matrix ndarray"
            )

        if isinstance(v, Ndarray):
            if v.layout != Layout.AOS:
                raise TaichiCompilationError(
                    f"AOT ndarray template argument {arg_name} must use "
                    "Layout.AOS"
                )
        elif isinstance(v, np.ndarray):
            if not v.flags.c_contiguous:
                raise TaichiCompilationError(
                    f"AOT external array template argument {arg_name} must "
                    "be C-contiguous"
                )
        elif kernel_impl.has_pytorch():
            import torch  # pylint: disable=C0415

            if not isinstance(v, torch.Tensor):
                raise TaichiCompilationError(
                    f"AOT ndarray template argument {arg_name} must be a "
                    "Taichi ndarray, NumPy ndarray, or Torch tensor"
                )
            if not v.is_contiguous():
                raise TaichiCompilationError(
                    f"AOT external array template argument {arg_name} must "
                    "be contiguous"
                )
        else:
            raise TaichiCompilationError(
                f"AOT ndarray template argument {arg_name} must be a Taichi "
                "ndarray or NumPy ndarray"
            )

        feature = kernel_impl.TaichiCallableTemplateMapper.extract_arg(
            v, anno, arg_name
        )
        element_type = feature[0]
        if not hasattr(element_type, "to_string"):
            element_type = kernel_impl.to_taichi_type(v.dtype)
        dtype_name = element_type.to_string()
        if hasattr(element_type, "get_shape"):
            element_shape = tuple(element_type.get_shape())
        elif hasattr(element_type, "shape") and callable(element_type.shape):
            element_shape = tuple(element_type.shape())
        else:
            element_shape = ()
        stride_bytes = _ti_core.data_type_size(element_type)
        shape_key = "x".join(str(x) for x in element_shape) or "scalar"
        return (
            "ndarray-"
            f"dtype_{KernelTemplate._key_atom(dtype_name)}-"
            f"ndim_{feature[1]}-element_shape_{shape_key}-layout_aos-"
            f"stride_bytes_{stride_bytes}-"
            f"needs_grad_{int(bool(feature[2]))}-"
            f"boundary_{int(feature[3])}"
        )

    def instantiate(self, **kwargs):
        name = self._kernel_fn.__name__
        kernel = self._kernel_fn._primal
        assert isinstance(kernel, kernel_impl.Kernel)
        reject_acceleration_structure_arguments(kernel, "AOT kernel templates")
        injected_args = []
        key_p = ""
        required = {
            arg.name
            for arg in kernel.arguments
            if isinstance(arg.annotation, (template, NdarrayType))
        }
        provided = set(kwargs)
        missing = sorted(required - provided)
        unknown = sorted(provided - required)
        if missing:
            raise TaichiCompilationError(
                "Missing AOT kernel template arguments: " + ", ".join(missing)
            )
        if unknown:
            raise TaichiCompilationError(
                "Unexpected AOT kernel template arguments: "
                + ", ".join(unknown)
            )

        for arg in kernel.arguments:
            if isinstance(arg.annotation, template):
                v = kwargs[arg.name]
                key_p += arg.name
                key_p = self.keygen(v, key_p, self._aot_module._fields.items())
                injected_args.append(v)
            elif isinstance(arg.annotation, NdarrayType):
                v = kwargs[arg.name]
                key_p += (
                    arg.name
                    + "="
                    + self._ndarray_keygen(v, arg.annotation, arg.name)
                    + "__"
                )
                injected_args.append(v)
            elif isinstance(arg.annotation, (TextureType, RWTextureType)):
                raise TaichiCompilationError(
                    "AOT kernel templates do not support texture "
                    f"specialization ({arg.name}); use Module.add_kernel() "
                    "with template_args"
                )
            else:
                injected_args.append(0)
        if len(key_p.encode("utf-8")) > 180:
            key_p = (
                "sha256_"
                + hashlib.sha256(key_p.encode("utf-8")).hexdigest()
            )
        if key_p in self._instantiated_keys:
            return
        kernel.ensure_compiled(*injected_args)
        self._aot_module._aot_builder.add_kernel_template(name, key_p, kernel.kernel_cpp)
        self._instantiated_keys.add(key_p)

        # kernel AOT
        self._aot_module._kernels.append(kernel)


class Module:
    """An AOT module to save and load Taichi kernels.

    This module serializes the Taichi kernels for a specific arch. The
    serialized module can later be loaded to run on that backend, without the
    Python environment.

    Example:
      Usage::

        m = ti.aot.Module(ti.metal)
        m.add_kernel(foo)
        m.add_kernel(bar)

        m.save('/path/to/module')

        # Now the module file '/path/to/module' contains the Metal kernels
        # for running ``foo`` and ``bar``.
    """

    def __init__(self, arch=None, caps=None):
        """Creates a new AOT module instance

        Args:
          arch: Target backend architecture. Default to the one initialized in :func:`~taichi_forge.lang.init` if not specified.
          caps (List[str]): Enabled device capabilities.
        """
        if caps is None:
            caps = []
        curr_arch = impl.current_cfg().arch
        if arch is None:
            arch = curr_arch
        elif arch != curr_arch:
            raise TaichiRuntimeError(
                "ti.aot.Module() currently supports same-target compilation "
                f"only: requested {arch}, but the active runtime is {curr_arch}. "
                "Initialize Taichi with the requested arch before creating the "
                "module; cross-arch compilation is not supported."
            )

        self._arch = arch
        self._kernels = []
        self._fields = {}
        rtm = impl.get_runtime()
        rtm._finalize_root_fb_for_aot()
        self._aot_builder = rtm.prog.make_aot_module_builder(arch, caps)
        self._content = []

    def add_field(self, name, field):
        """Add a taichi field to the AOT module.

        Args:
          name: name of taichi field
          field: taichi field

        Example::

            >>> a = ti.field(ti.f32, shape=(4,4))
            >>> b = ti.field("something")
            >>>
            >>> m.add_field(a)
            >>> m.add_field(b)
            >>>
            >>> # Must add in sequence
        """
        is_scalar = True
        self._fields[name] = field
        column_num = 1
        row_num = 1
        if isinstance(field, MatrixField):
            is_scalar = False
            row_num = field.m
            column_num = field.n
        else:
            assert isinstance(field, ScalarField)
        self._aot_builder.add_field(
            name,
            field.snode.ptr,
            is_scalar,
            field.dtype,
            field.snode.shape,
            row_num,
            column_num,
        )

    def add_kernel(self, kernel_fn, template_args=None, name=None):
        """Add a taichi kernel to the AOT module.

        Args:
          kernel_fn (Function): the function decorated by taichi `kernel`.
          template_args (Dict[str, Any]): a dict where key is the template
            parameter name, and value is the instantiating arg. Note that this
            works for both :class:`~taichi_forge.types.template` and for
            `:class:`~taichi_forge.types.ndarray`.
          name (str): Name to identify this kernel in the module. If not
            provided, uses the built-in ``__name__`` attribute of `kernel_fn`.

        """
        kernel_name = name or kernel_fn.__name__
        kernel = kernel_fn._primal
        assert isinstance(kernel, kernel_impl.Kernel)
        reject_acceleration_structure_arguments(kernel, "AOT Module.add_kernel()")
        if template_args is not None:
            injected_args = produce_injected_args_from_template(kernel, template_args)
        else:
            injected_args = produce_injected_args(kernel)
        kernel.ensure_compiled(*injected_args)
        self._aot_builder.add(kernel_name, kernel.kernel_cpp)

        # kernel AOT
        self._kernels.append(kernel)

        self._content += ["kernel:" + kernel_name]

    def add_graph(self, name, graph):
        if getattr(graph, "_contains_native_nodes", False):
            raise TaichiRuntimeError(
                "AOT Module.add_graph() does not serialize Forge native graph "
                "nodes. Native graph replay is JIT-only and limited to "
                "DSL-defined native methods; AOT graph export currently "
                "supports ordinary kernel CGraphs only."
            )
        self._aot_builder.add_graph(name, graph._compiled_graph)
        self._content += ["cgraph:" + name]

    @contextmanager
    def add_kernel_template(self, kernel_fn):
        """Add a taichi kernel (with template parameters) to the AOT module.

        Args:
          kernel_fn (Function): the function decorated by taichi `kernel`.

        Example::

            >>> @ti.kernel
            >>> def bar_tmpl(a: ti.template()):
            >>>   x = a
            >>>   # or y = a
            >>>   # do something with `x` or `y`
            >>>
            >>> m = ti.aot.Module(arch)
            >>> with m.add_kernel_template(bar_tmpl) as kt:
            >>>   kt.instantiate(a=x)
            >>>   kt.instantiate(a=y)
            >>>
            >>> @ti.kernel
            >>> def bar_tmpl_multiple_args(a: ti.template(), b: ti.template())
            >>>   x = a
            >>>   y = b
            >>>   # do something with `x` and `y`
            >>>
            >>> with m.add_kernel_template(bar_tmpl) as kt:
            >>>   kt.instantiate(a=x, b=y)

        Ndarray parameters may be specialized with a Taichi ndarray, a
        C-contiguous NumPy array, or a contiguous Torch tensor exemplar.
        Specialization keys describe element/layout ABI but do not include
        runtime shape extents.
        """
        kt = KernelTemplate(kernel_fn, self)
        yield kt

    def save(self, filepath):
        """
        Args:
          filepath (str): path to a folder to store aot files.
        """
        filepath = str(PurePosixPath(Path(filepath)))
        self._aot_builder.dump(filepath, "")
        with open(f"{filepath}/__content__", "w") as f:
            f.write("\n".join(self._content))
        with open(f"{filepath}/__version__", "w") as f:
            f.write(".".join(str(x) for x in taichi_forge.__version__))

    def archive(self, filepath: str):
        """
        Args:
          filepath (str): path to the stored archive of aot artifacts, MUST
            end with `.tcm`.
        """
        assert filepath.endswith(".tcm"), "AOT module artifact archive must ends with .tcm"
        tcm_path = Path(filepath).absolute()
        assert tcm_path.parent.exists(), "Output directory doesn't exist"

        temp_dir = mkdtemp(prefix="tcm_")
        # Save first as usual.
        self.save(temp_dir)

        fixed_time = datetime.datetime(2000, 12, 1).timestamp()

        # Package all artifacts into a zip archive and attach contend data.
        with ZipFile(tcm_path, "w") as z:
            for path in glob(f"{temp_dir}/*", recursive=True):
                os.utime(path, (fixed_time, fixed_time))
                z.write(path, Path.relative_to(Path(path), temp_dir))

        # Remove cached files
        rmtree(temp_dir)
