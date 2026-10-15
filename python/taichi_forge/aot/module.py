import datetime
import hashlib
import os
from contextlib import contextmanager
from pathlib import Path
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
    reject_texture_collection_arguments,
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


def _resolve_kernel(kernel_fn):
    """Resolve an export target without silently switching an adjoint to primal."""
    bound = None
    if isinstance(kernel_fn, kernel_impl.Kernel):
        kernel = kernel_fn
    elif isinstance(kernel_fn, kernel_impl._BoundedDifferentiableMethod):
        bound = kernel_fn
        kernel = bound._primal
    elif (
        isinstance(getattr(kernel_fn, "__self__", None), kernel_impl._BoundedDifferentiableMethod)
        and kernel_fn.__func__ is kernel_impl._BoundedDifferentiableMethod.grad
    ):
        bound = kernel_fn.__self__
        kernel = bound._adjoint
    else:
        kernel = getattr(kernel_fn, "_primal", None)
    if not isinstance(kernel, kernel_impl.Kernel):
        raise TypeError("AOT export expects a Taichi kernel or kernel.grad")
    name = kernel.func.__name__
    if kernel.autodiff_mode == _ti_core.AutodiffMode.REVERSE:
        name += "_grad"
    bound_args = {}
    if bound is not None and not bound._is_staticmethod:
        bound_args[kernel.arguments[0].name] = bound._kernel_owner
    return kernel, name, bound_args


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
        if isinstance(v, StructNdarray):
            # Avoid exporting unbounded filesystem names for nested structures.
            # The full structural spelling remains the hash input.
            dtype_name = "struct_" + hashlib.sha256(dtype_name.encode()).hexdigest()
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
        kernel, name, bound_args = _resolve_kernel(self._kernel_fn)
        reject_acceleration_structure_arguments(kernel, "AOT kernel templates")
        reject_texture_collection_arguments(kernel, "AOT kernel templates")
        injected_args = []
        key_p = ""
        required = {
            arg.name
            for arg in kernel.arguments
            if isinstance(arg.annotation, (template, NdarrayType))
        } - set(bound_args)
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
            if arg.name in bound_args:
                injected_args.append(bound_args[arg.name])
            elif isinstance(arg.annotation, template):
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
        key = kernel.ensure_compiled(*injected_args, _for_aot=True)
        compiled = kernel.compiled_kernels[key]
        export_name = name + "__tmpl__" + key_p
        if self._aot_module._check_kernel_export(export_name, compiled):
            self._aot_module._aot_builder.add_kernel_template(name, key_p, compiled)
            self._aot_module._exported_kernels[export_name] = compiled
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

    def __init__(self, arch=None, caps=None, *, compile_options=None, artifact_format=None):
        """Creates a new AOT module instance

        Args:
          arch: Target backend architecture. Default to the one initialized in :func:`~taichi_forge.lang.init` if not specified.
          caps (List[str]): Enabled device capabilities.
          compile_options (dict): Optional export-only overrides for
            ``compile_tier``, ``advanced_optimization``, ``llvm_opt_level``,
            ``external_optimization_level``, ``fast_math`` and
            ``full_simplify_global_iter_cap`` on CPU/CUDA/Vulkan. Explicit
            entries override per-kernel and Program values; omitted entries
            inherit them. Options are copied and do not mutate JIT settings.
          artifact_format (str): Optional ``bc`` for CPU/CUDA LLVM bitcode or
            ``ptx`` or ``cubin`` for CUDA device code. PTX moves user-kernel LLVM compilation
            to export; driver compilation and runtime initialization remain.
            CUDA target requirements are selected by ``caps``.
            Cubin requires an explicit ``cuda_compute_capability`` capability
            and external ptxas at export (``TI_CUDA_PTXAS_PATH`` or PATH). It
            stores machine code without implicit PTX fallback. Runtime bootstrap
            still uses LLVM; binary target compatibility is checked at load.
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

        if artifact_format is not None:
            if artifact_format not in ("bc", "ptx", "cubin"):
                raise ValueError("AOT artifact_format must be bc, ptx or cubin")
            if artifact_format in ("ptx", "cubin") and arch != taichi_forge.cuda:
                raise TaichiRuntimeError(f"AOT {artifact_format.upper()} artifacts require CUDA")
            if artifact_format == "bc" and arch not in (taichi_forge.cpu, taichi_forge.cuda):
                raise TaichiRuntimeError("AOT bitcode artifacts require CPU or CUDA")

        options = None
        if compile_options is not None:
            if not isinstance(compile_options, dict):
                raise TypeError("AOT compile_options must be a dict")
            if compile_options and arch not in (taichi_forge.cpu, taichi_forge.cuda, taichi_forge.vulkan):
                raise TaichiRuntimeError("AOT compile_options currently supports CPU, CUDA and Vulkan")
            allowed = {
                "compile_tier", "advanced_optimization", "llvm_opt_level",
                "external_optimization_level", "fast_math", "full_simplify_global_iter_cap",
            }
            unknown = set(compile_options) - allowed
            if unknown:
                raise ValueError("Unknown AOT compile options: " + ", ".join(sorted(unknown)))
            options = _ti_core.KernelCompileOptions()
            for key, value in compile_options.items():
                if key == "compile_tier" and value is not None and value not in ("fast", "balanced", "full"):
                    raise ValueError("AOT compile_tier must be fast, balanced or full")
                if key in ("llvm_opt_level", "external_optimization_level") and value is not None:
                    if not isinstance(value, int) or not 0 <= value <= 3:
                        raise ValueError(f"AOT {key} must be an integer from 0 to 3")
                setattr(options, key, value)

        self._arch = arch
        self._kernels = []
        self._exported_kernels = {}
        self._execution_plans = {}
        self._data_payloads = {}
        self._initial_root_data = {}
        self._graph_export_failed = False
        self._fields = {}
        self._array_fields = set()
        rtm = impl.get_runtime()
        rtm._finalize_root_fb_for_aot()
        self._aot_builder = rtm.prog.make_aot_module_builder(arch, caps)
        if artifact_format is not None:
            self._aot_builder.set_artifact_format(artifact_format)
        if options is not None:
            self._aot_builder.set_compile_options(options)
        self._content = []

    def add_field(self, name, field, *, as_array=False):
        """Add a taichi field to the AOT module.

        Args:
          name: name of taichi field
          field: taichi field
          as_array: Also expose packed dense storage as a named ``module_array``
            execution resource on CPU/CUDA/Vulkan. The view shares module state;
            LLVM exports require relocatable dense trees. Defaults to False.

        Example::

            >>> a = ti.field(ti.f32, shape=(4,4))
            >>> b = ti.field("something")
            >>>
            >>> m.add_field(a)
            >>> m.add_field(b)
            >>>
            >>> # Must add in sequence
        """
        storage = None
        if not isinstance(as_array, bool):
            raise TypeError("AOT as_array must be a bool")
        if as_array or name in self._array_fields:
            if not isinstance(name, str) or not name or "\0" in name:
                raise ValueError("AOT module array names must be nonempty strings without NUL")
            if name in self._fields and self._fields[name] is not field:
                raise ValueError("AOT module array names cannot be rebound to another Field")
        if as_array:
            if self._arch not in (_ti_core.x64, _ti_core.arm64, _ti_core.cuda, _ti_core.vulkan):
                raise ValueError("AOT module array views require CPU, CUDA or Vulkan")
            from taichi_forge.lang._storage_view import describe_storage

            storage = describe_storage(field)
            if not storage.supported or not all(
                storage.properties[key] for key in ("compact_contiguous", "ndarray_abi_compatible", "aligned")
            ):
                raise ValueError("AOT module array views require aligned packed dense Field storage")
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
        if storage is not None:
            self._aot_builder._add_root_storage_view(name, storage.descriptor)
            self._array_fields.add(name)

    def _check_kernel_export(self, name, compiled):
        previous = self._exported_kernels.get(name)
        if previous is not None and previous is not compiled:
            raise TaichiCompilationError(f"AOT kernel name {name!r} is already bound to another specialization")
        return previous is None

    def add_kernel(self, kernel_fn, template_args=None, name=None):
        """Add a taichi kernel to the AOT module.

        Args:
          kernel_fn (Function): a Taichi kernel or its explicit ``.grad`` kernel.
          template_args (Dict[str, Any]): a dict where key is the template
            parameter name, and value is the instantiating arg. Note that this
            works for both :class:`~taichi_forge.types.template` and for
            `:class:`~taichi_forge.types.ndarray`.
          name (str): Name to identify this kernel in the module. If not
            provided, uses the source function name, suffixed with ``_grad``
            for a reverse kernel. A bound method freezes its owner at export.

        Dense-field reverse kernels use the same backend autodiff restrictions
        as JIT. Export explicit initialization, gradient clearing/seeding and
        readback kernels; this does not serialize Tape orchestration.

        """
        kernel, default_name, bound_args = _resolve_kernel(kernel_fn)
        kernel_name = name or default_name
        reject_acceleration_structure_arguments(kernel, "AOT Module.add_kernel()")
        reject_texture_collection_arguments(kernel, "AOT Module.add_kernel()")
        if template_args is not None:
            if set(template_args) & set(bound_args):
                raise TaichiCompilationError("Cannot override the owner of a bound AOT kernel")
            template_args = {**template_args, **bound_args}
            injected_args = produce_injected_args_from_template(kernel, template_args)
        else:
            # Bound self is already a compile-time value. Do not synthesize a
            # scalar placeholder for its ti.template annotation first.
            injected_args = produce_injected_args(kernel, bound_args=bound_args)
        key = kernel.ensure_compiled(*injected_args, _for_aot=True)
        compiled = kernel.compiled_kernels[key]
        if not self._check_kernel_export(kernel_name, compiled):
            return
        self._aot_builder.add(kernel_name, compiled)
        self._exported_kernels[kernel_name] = compiled

        # kernel AOT
        self._kernels.append(kernel)

        self._content += ["kernel:" + kernel_name]

    def add_graph(self, name, graph):
        if getattr(graph, "_contains_native_nodes", False):
            raise TaichiRuntimeError(
                "AOT Module.add_graph() does not serialize Forge native graph "
                "nodes into the CGraph format. Use Module.add_execution_plan(name, graph) "
                "for recordings with an AOT reconstruction contract."
            )
        self._aot_builder.add_graph(name, graph._compiled_graph)
        self._content += ["cgraph:" + name]

    def add_data(self, data):
        """Snapshot packed bytes for ndarray initial_data; return their SHA-256 key.

        Identical payloads share one artifact entry. Only payloads referenced by
        exported plans are saved. Device arrays remain private to each preparation.
        """
        snapshot = memoryview(data).tobytes()
        key = hashlib.sha256(snapshot).hexdigest()
        previous = self._data_payloads.setdefault(key, snapshot)
        if previous != snapshot:
            raise ValueError("AOT data digest collision")
        return key

    def _add_initial_root_snapshots(self, dependencies, payloads):
        """Publish one coherent root snapshot after its export closure is known."""
        if len(dependencies) != len(payloads):
            raise ValueError("AOT root snapshot count does not match dependencies")
        pending = {}
        data = {}
        for (tree, generation, layout), payload in zip(dependencies, payloads):
            snapshot = bytes(payload)
            key = hashlib.sha256(snapshot).hexdigest()
            descriptor = (generation, layout, len(snapshot), key)
            existing = pending.get(tree, self._initial_root_data.get(tree))
            if existing is not None and existing != descriptor:
                raise ValueError(f"AOT initial root state conflicts for tree {tree}; create a new module")
            previous = data.get(key, self._data_payloads.get(key))
            if previous is not None and previous != snapshot:
                raise ValueError("AOT data digest collision")
            pending[tree] = descriptor
            data[key] = snapshot
        self._aot_builder._add_initial_root_data(
            [(tree, descriptor[2], descriptor[3]) for tree, descriptor in pending.items()]
        )
        self._initial_root_data.update(pending)
        self._data_payloads.update(data)

    def add_execution_plan(self, name, plan, *, gradients=None, snapshot_state=False, source_bindings=None):
        """Export ordered loaded-kernel/copy/fill commands with symbolic storage.

        An ExecutionPlan requires its kernels to be exported first. A compiled
        Graph instead exports its referenced kernels and supported native
        recordings automatically; optional gradients map primal array names to
        their gradient bindings. Unsupported control/resource recipes fail before
        publication. The resulting plan owns no live Python callback,
        Graph definition or device handle; the consumer binds arrays and records
        native execution at its cold boundary.

        ``snapshot_state=True`` snapshots qualified provider workspace and dense
        Field roots at export. Provider workspace is private to each prepared
        consumer; Field state is shared by all entries of one loaded module and
        initialized once. Separate module instances isolate Field state. Changes
        to an already exported root snapshot require a new producer module.

        ``source_bindings`` maps public names to scalar producer ndarrays held
        by qualified native operations. Export freezes type and shape, without
        copying data or storing addresses; consumers supply their own arrays.
        Reuse a kernel's symbolic name when it accesses the same storage.
        """
        from taichi_forge_runtime.aot import ExecutionPlan
        from taichi_forge.graph._graph import Graph

        if not isinstance(name, str) or not name or "\0" in name:
            raise ValueError("Execution plan name must be a nonempty string without NUL")
        if type(snapshot_state) is not bool:
            raise TypeError("snapshot_state must be a bool")
        if self._graph_export_failed:
            raise TaichiRuntimeError("A Graph kernel export failed; create a new AOT module")
        if name in self._execution_plans:
            raise ValueError("Duplicate execution plan export: " + name)
        if isinstance(plan, Graph):
            from taichi_forge.aot._graph_export import export_graph

            plan = export_graph(
                self, plan, gradients=gradients, snapshot_state=snapshot_state, source_bindings=source_bindings
            )
        elif not isinstance(plan, ExecutionPlan):
            raise TypeError("Expected an ExecutionPlan or compiled Graph")
        elif gradients is not None:
            raise ValueError("Specify gradients on the ExecutionPlan itself")
        elif snapshot_state:
            raise ValueError("snapshot_state requires a compiled Graph")
        elif source_bindings is not None:
            raise ValueError("source_bindings requires a compiled Graph")
        snapshot = ExecutionPlan.from_json(plan.to_json())
        missing = set(snapshot.kernel_entries) - self._exported_kernels.keys()
        if missing:
            raise ValueError("Execution plan references unexported kernels: " + ", ".join(sorted(missing)))
        missing_data = set(snapshot.data_entries) - self._data_payloads.keys()
        if missing_data:
            raise ValueError("Execution plan references unexported data: " + ", ".join(sorted(missing_data)))
        self._execution_plans[name] = snapshot
        self._content.append("execution:" + name)
        return snapshot

    def add_primitive(self, name, kind, *, version=1, **options):
        """Export a registered primitive recipe and its selected helper kernels.

        Version 1 provides ``sort``, ``compact`` and consecutive ``unique``.
        Options name symbolic array bindings; unique additionally accepts
        ``key_dtype`` (i32/u32/i64/u64) and optional ``indices`` output. Unique
        processes the entire bound key extent, keeps the first item of each
        run, and does not sort. See the AOT guide for payload and count bindings.

        Backend support comes from the shared native factory registry. Export
        does not allocate device scratch or execute the primitive. Preparation
        validates actual storage and reconstructs private resources. Returns
        the frozen ExecutionPlan, also saved under ``name`` in this module.
        """
        from taichi_forge.aot._primitive_exports import export_primitive

        return export_primitive(self, name, kind, version, options)

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
        if self._graph_export_failed:
            raise TaichiRuntimeError("A Graph kernel export failed; create a new AOT module")
        output_dir = Path(filepath)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir.as_posix()
        self._aot_builder.dump(filepath, "")
        data_entries = {key for plan in self._execution_plans.values() for key in plan.data_entries}
        data_entries.update(descriptor[3] for descriptor in self._initial_root_data.values())
        if data_entries:
            directory = output_dir / "data"
            directory.mkdir(exist_ok=True)
            for key in sorted(data_entries):
                (directory / (key + ".bin")).write_bytes(self._data_payloads[key])
        if self._execution_plans:
            directory = output_dir / "executions"
            directory.mkdir(exist_ok=True)
            for name, plan in self._execution_plans.items():
                plan.save(directory / (quote(name, safe="-_.") + ".json"))
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
        if self._graph_export_failed:
            raise TaichiRuntimeError("A Graph kernel export failed; create a new AOT module")
        tcm_path = Path(filepath).absolute()
        assert tcm_path.parent.exists(), "Output directory doesn't exist"

        temp_dir = mkdtemp(prefix="tcm_")
        # Save first as usual.
        self.save(temp_dir)

        fixed_time = datetime.datetime(2000, 12, 1).timestamp()

        # Package all artifacts into a zip archive and attach contend data.
        with ZipFile(tcm_path, "w") as z:
            for path in sorted(Path(temp_dir).rglob("*")):
                if not path.is_file():
                    continue
                os.utime(path, (fixed_time, fixed_time))
                z.write(path, Path.relative_to(Path(path), temp_dir))

        # Remove cached files
        rmtree(temp_dir)
