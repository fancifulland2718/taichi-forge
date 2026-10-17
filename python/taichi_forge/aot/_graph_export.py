"""Cold Graph lowering into the shared, pointer-free execution description."""

import ctypes
import hashlib
import json
import struct

from taichi_forge._lib import core
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge_runtime.aot import ExecutionPlan, Scalar


def _snapshot(value):
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


class _ExportContext:
    """One export transaction; object identities never become artifact data."""

    def __init__(self, module, reserved_names, *, snapshot_state=False, source_arrays=None, source_resources=None, resource_inputs=None):
        self.arch = module._arch
        self.snapshot_state = snapshot_state
        self.source_arrays = source_arrays
        self.source_resources = source_resources
        self.source_images = source_resources.images if source_resources is not None else None
        self.resource_inputs = resource_inputs
        self.operations = []
        self.initialization = []
        self._initialization_resources = {}
        self.data_payloads = {}
        self._data_keys = {}
        self._readonly_data_owners = {}
        self.resources = []
        self.kernels = {}
        self.kernel_launch_abis = {}
        self._used_kernel_entries = set()
        self._state_dependencies = {}
        self.root_dependencies = ()
        self.root_payloads = ()
        self._resource_owners = {}
        self._resource_policies = {}
        self._public_resource_owners = {}
        self._resource_links = {}
        self._indices = {}
        self._names = set(reserved_names) | set(module._exported_kernels) | set(module._fields)
        self._kernel_names = {
            (id(kernel), module._kernel_launch_abis.get(name, "ordinary")): name
            for name, kernel in module._exported_kernels.items()
        }
        self.reconstructed_binding_validators = set()
        self.reconstructed_argument_binders = set()
        self.reconstructed_derived_bindings = set()
        self.binding_constraints = []
        self._provider_data = {}
        self.bounded_extents = {}
        self.private_bindings = {}
        self._private_names = {}
        self._scalar_values = {}
        self._scalar_symbols = {}
        self._scalar_resources = {}
        self.control_names = set()
        self._control_bindings = set()
        from ._graph_storage_bindings import RetainedArrayBindings

        self.retained_arrays = RetainedArrayBindings(module, self)

    def region(self, emit, node, **kwargs):
        """Collect a selected region without copying its code/resource owners."""
        outer, self.operations = self.operations, []
        try:
            emit(self, node, **kwargs)
            return self.operations
        finally:
            self.operations = outer

    def _name(self, kind):
        index = self._indices.get(kind, 0)
        while (name := f"__forge_aot_{kind}_{index}") in self._names:
            index += 1
        self._names.add(name)
        self._indices[kind] = index + 1
        return name

    def resource(self, owner, provider, operation, *, parameters, bindings=None, version=1):
        return self._resource(
            owner,
            dict(
                kind="native",
                provider=provider,
                operation=operation,
                version=version,
                parameters=parameters,
                bindings={port: self.binding(slot, resource=True) for port, slot in (bindings or {}).items()},
            ),
        )

    def resource_once(self, owner, build, *, policy=None):
        """Sample a qualified live resource once within this export transaction."""
        previous = self._resource_owners.get(id(owner))
        if previous is not None:
            if self._resource_policies.get(id(owner)) != policy:
                raise ValueError("AOT resource reconstruction policy changed within one export")
            return previous[1]
        name = build()
        created = self._resource_owners.get(id(owner))
        if created is None or created[1] != name:
            raise ValueError("AOT resource factory did not register its owner")
        self._resource_policies[id(owner)] = policy
        return name

    def public_resource(self, owner):
        name = self.resource_inputs.public(owner)
        if name is not None:
            self._public_resource_owners[id(owner)] = (owner, name)
        return name

    def resource_links(self, owner, links):
        """Record qualified source relationships; never serialize source handles."""
        links = tuple(sorted(links.items()))
        previous = self._resource_links.get(id(owner))
        if previous is not None:
            if len(previous[1]) != len(links) or any(
                old_port != port or old_child is not child
                for (old_port, old_child), (port, child) in zip(previous[1], links)
            ):
                raise ValueError("Native resource relationships changed during export")
        else:
            self._resource_links[id(owner)] = (owner, links)

    def resolve_resource_aliases(self):
        # Visit each source resource's children once; repeated occurrences emit
        # only an identity certificate instead of expanding a shared DAG again.
        canonical = {}
        for owner, name in self._public_resource_owners.values():
            stack = [(owner, ())]
            while stack:
                resource, path = stack.pop()
                key = id(resource)
                if key in self._resource_owners:
                    raise ValueError(
                        "A public resource dependency is also rebuilt privately; bind the shared dependency publicly"
                    )
                if key in canonical:
                    left, left_path = canonical[key]
                    if (left, left_path) != (name, path):
                        self.binding_constraints.append(
                            dict(
                                kind="resource_alias",
                                left=left,
                                right=name,
                                left_path=list(left_path),
                                right_path=list(path),
                            )
                        )
                    continue
                canonical[key] = (name, path)
                if key in self._resource_links:
                    stack.extend((child, path + (port,)) for port, child in self._resource_links[key][1])

    def _resource(self, owner, description, *, name=None):
        """Share only an identical producer resource, with an identical description."""
        if owner is None:
            raise ValueError("AOT reconstruction requires a resource owner identity")
        description = _snapshot(description)
        previous = self._resource_owners.get(id(owner))
        if previous is not None:
            _, name, frozen = previous
            if frozen != description:
                raise ValueError("AOT resource reconstruction changed within one export")
            return name
        name = name or self._name("resource")
        self._resource_owners[id(owner)] = (owner, name, description)
        self.resources.append(dict(description, name=name))
        return name

    def declare_private_bindings(self, bindings):
        from taichi_forge.graph._ir import InternalNdarrayRequirement
        from taichi_forge.lang.util import cook_dtype

        for name, owner in bindings.items():
            if self.source_resources is not None and name in self.source_resources.owners:
                if self.source_resources.owners[name] is not owner:
                    raise TaichiRuntimeError("AOT private resource changed within one export")
                self.private_bindings[name] = owner
                continue
            if type(owner) in (bool, int, float):
                # The final spec value is the shared binding frame. Source
                # nodes may retain equal but nonidentical earlier values.
                self.private_bindings[name] = self._scalar_values[name] = owner
                continue
            if self.source_arrays is not None and (external := self.source_arrays.find(owner)) is not None:
                self.private_bindings[name] = owner
                self._private_names[name] = external
                continue
            if not isinstance(owner, InternalNdarrayRequirement):
                continue
            dtype = cook_dtype(owner.dtype)
            element_type = str(dtype.element_type())
            if element_type not in {"f16", "f32", "f64", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"}:
                raise TaichiRuntimeError("AOT private ndarray requires a supported scalar element type")
            if core.data_type_size(dtype) != owner.element_bytes:
                raise TaichiRuntimeError("AOT private ndarray element size differs from its declaration")
            self._private_names[name] = self._resource(
                owner,
                dict(
                    kind="ndarray",
                    dtype=element_type,
                    shape=owner.shape,
                    element_shape=dtype.shape(),
                    zero_initialize=True,
                ),
            )
            self.private_bindings[name] = owner

    def data_resource(self, owner, data, *, dtype, shape, element_shape=(), name=None, readonly=False):
        """Stage already frozen data; never sample a live owner during export."""
        if type(data) is not bytes:
            raise TypeError("AOT resource data must be an immutable byte snapshot")
        previous = self._data_keys.get(id(data))
        if previous is None:
            key = hashlib.sha256(data).hexdigest()
            if key in self.data_payloads and self.data_payloads[key] != data:
                raise ValueError("AOT data digest collision")
            self.data_payloads[key] = data
            self._data_keys[id(data)] = (data, key)
        else:
            key = previous[1]
        if readonly:
            # Qualified exporters must never pass shared data to a writable
            # operation or expose it as mutable state.
            if name is not None:
                raise ValueError("Shared read-only AOT data cannot reserve a private binding name")
            identity = key, dtype, tuple(shape), tuple(element_shape)
            owner = self._readonly_data_owners.setdefault(identity, owner)
        return self._resource(
            owner,
            dict(kind="ndarray", dtype=dtype, shape=shape, element_shape=element_shape, initial_data=key),
            name=name,
        )

    def host_blob_resource(self, data):
        """Retain qualified immutable provider code/data without device storage."""
        if type(data) is not bytes:
            raise TypeError("AOT host blobs must be immutable bytes")
        key = hashlib.sha256(data).hexdigest()
        previous = self.data_payloads.setdefault(key, data)
        if previous != data:
            raise ValueError("AOT data digest collision")
        owner = self._readonly_data_owners.setdefault(("host_blob", key), object())
        return self._resource(owner, dict(kind="host_blob", data=key, byte_size=len(data)))

    def binding(self, name, *, resource=False):
        if self.source_resources is not None and name in self.source_resources.owners:
            owner = self.source_resources.owners[name]
            qualified = self._public_resource_owners.get(id(owner))
            if not resource or qualified is None or qualified[0] is not owner or qualified[1] != name:
                raise TaichiRuntimeError("AOT resource bindings require a qualified typed operation")
        if name in self._scalar_values:
            raise TaichiRuntimeError("AOT fixed scalar requires a typed kernel argument")
        return self._private_names.get(name, name)

    def view_binding(self, name, parent, offset, shape, *, owner):
        if name in self.private_bindings and self.private_bindings[name] is not owner:
            raise TaichiRuntimeError("AOT derived view conflicts with an existing private binding")
        self._private_names[name] = self._resource(
            owner, dict(kind="ndarray_view", array=self.binding(parent), offset=offset, shape=shape)
        )
        self.private_bindings[name] = owner
        self.reconstructed_derived_bindings.add(name)

    def provider_data_binding(self, name, owner, *, mutable=False):
        """Declare qualified provider storage; sample the complete closure once."""
        from taichi_forge.graph._native import ProviderOwnedNdarrayBinding
        from taichi_forge.lang import impl

        if mutable and not self.snapshot_state:
            raise TaichiRuntimeError("AOT provider workspace requires explicit snapshot_state=True at export")
        if type(owner) is not ProviderOwnedNdarrayBinding or owner._runtime_prog is not impl.get_runtime().prog:
            raise TaichiRuntimeError("AOT provider data requires a live qualified generation binding")
        identity = owner._runtime_allocation_identity
        cached = self._provider_data.get(identity)
        array = owner.arr
        dtype, shape = str(array.element_data_type()), tuple(array.shape)
        if tuple(array.total_shape()) != shape:
            raise TaichiRuntimeError("AOT provider data currently requires scalar arrays")
        if cached is None:
            resource = self._name("resource")
            self._provider_data[identity] = (owner, resource, dtype, shape)
        else:
            _, resource, previous_dtype, previous_shape = cached
            if (dtype, shape) != (previous_dtype, previous_shape):
                raise TaichiRuntimeError("AOT provider allocation changed its data description")
        if name in self.private_bindings and self.private_bindings[name] is not owner:
            raise TaichiRuntimeError("AOT provider data binding changed within one export")
        self.private_bindings[name] = owner
        self._private_names[name] = resource

    def field_state(self, dependencies):
        """Retain source lifetime proofs; the compiled AOT closure selects bytes."""
        for dependency in dependencies:
            if not self.snapshot_state:
                raise TaichiRuntimeError("AOT live Field state requires explicit snapshot_state=True at export")
            dependency = tuple(dependency)
            previous = self._state_dependencies.setdefault(dependency[0], dependency)
            if previous != dependency:
                raise TaichiRuntimeError("AOT Graph has conflicting Field lifetime dependencies")

    def materialize_provider_data(self, module):
        """Sample exact compiled roots and selected provider arrays together."""
        import numpy as np

        from taichi_forge.lang import impl
        from taichi_forge.lang.util import to_numpy_type

        pending = tuple(self._provider_data.values())
        arrays = [owner.arr for owner, _, _, _ in pending]
        program = impl.get_runtime().prog
        ids = sorted(
            set(module._aot_builder._used_snode_tree_ids(sorted(self._used_kernel_entries)))
            | self.retained_arrays.tree_ids
        )
        dependencies = tuple(program._snapshot_snode_tree_dependencies(ids))
        if dependencies:
            if not self.snapshot_state:
                raise TaichiRuntimeError("AOT compiled Field state requires explicit snapshot_state=True at export")
            for dependency in dependencies:
                expected = self._state_dependencies.get(dependency[0])
                if expected is not None and expected != tuple(dependency):
                    raise TaichiRuntimeError("AOT compiled Field lifetime changed during export")
            snapshot = program._snapshot_dense_state(dependencies, arrays)
            self.root_dependencies, self.root_payloads = dependencies, snapshot["roots"]
            for (owner, name, dtype, shape), data in zip(pending, snapshot["arrays"]):
                self.data_resource(owner, data, dtype=dtype, shape=shape, name=name)
            return
        if not pending:
            return
        hosts = [
            np.empty(shape, dtype=to_numpy_type(array.element_data_type()))
            for array, (_, _, _, shape) in zip(arrays, pending)
        ]
        program.copy_ndarrays_to_host(arrays, hosts, snapshot=True)
        for (owner, name, dtype, shape), host in zip(pending, hosts):
            self.data_resource(owner, host.tobytes(), dtype=dtype, shape=shape, name=name)

    def kernel_binding(self, arg):
        if arg.tag in (core.ArgKind.TEXTURE, core.ArgKind.RWTEXTURE, core.ArgKind.ACCELERATION_STRUCTURE):
            return self.source_resources.kernel_resource(self, arg)
        if arg.name in self.control_names:
            if arg.tag != core.ArgKind.NDARRAY or str(arg.dtype()) != "i32":
                raise TaichiRuntimeError("AOT Graph control requires scalar i32 ndarray arguments")
            self._control_bindings.add(arg.name)
            self.binding_constraints.append(
                dict(
                    kind="array_layout",
                    array=self.binding(arg.name),
                    exact_shape=[1] * arg.field_dim,
                    element_shape=[],
                    record_bytes=4,
                    alignment=4,
                )
            )
        if arg.name not in self._scalar_values:
            return self.binding(arg.name)
        if arg.tag != core.ArgKind.SCALAR:
            raise TaichiRuntimeError("AOT fixed scalar is used as a non-scalar argument")
        dtype = str(arg.dtype())
        symbol = (arg.name, dtype)
        if symbol in self._scalar_symbols:
            return self._scalar_symbols[symbol]
        value = self._scalar_values[arg.name]
        try:
            if dtype in ("f16", "f32"):
                # F32 matches Graph's C++ cast, including overflow to infinity;
                # F16 also uses a float32 payload in the AOT scalar ABI.
                value = ctypes.c_float(float(value)).value
            resource = Scalar(dtype, value).resource(arg.name)
        except (ValueError, TypeError, OverflowError, struct.error) as error:
            raise TaichiRuntimeError(f"AOT fixed scalar {arg.name!r} cannot be encoded as {dtype}") from error
        identity = (dtype, resource["bits"])
        name = self._scalar_resources.get(identity)
        if name is None:
            name = self._name("resource")
            self._scalar_resources[identity] = name
            resource["name"] = name
            self.resources.append(resource)
        self._scalar_symbols[symbol] = name
        return name

    def declared_bindings(self, bindings):
        return {
            name: owner
            for name, owner in bindings.items()
            if name in self.private_bindings
            and (
                self.private_bindings[name] is owner
                or (
                    name in self._scalar_values
                    and type(owner) in (bool, int, float)
                    and owner == self._scalar_values[name]
                )
            )
        }

    def require_private_bindings(self, bindings):
        if len(self.declared_bindings(bindings)) != len(bindings):
            raise TaichiRuntimeError(
                "AOT fixed bindings require a reconstruction description; live arrays and unsupported values are not declarations"
            )

    def native(self, provider, operation, *, bindings, parameters=None, version=1):
        self.operations.append(self._native_description(provider, operation, bindings, parameters, version))

    def _native_description(self, provider, operation, bindings, parameters, version):
        return _snapshot(
            dict(
                kind="native",
                provider=provider,
                operation=operation,
                version=version,
                parameters=parameters or {},
                bindings={port: self.binding(slot, resource=True) for port, slot in bindings.items()},
            )
        )

    def initialize_native_once(self, resource, provider, operation, *, bindings, parameters=None, version=1):
        """One consistent setup per shared resource, outside replay."""
        description = self._native_description(provider, operation, bindings, parameters, version)
        previous = self._initialization_resources.get(resource)
        if previous is not None:
            if previous != description:
                raise ValueError("AOT resource initialization changed within one export")
            return
        self._initialization_resources[resource] = description
        self.initialization.append(description)

    def kernel(self, kernel, args, label="", launch=None, *, launch_abi="ordinary"):
        if launch_abi != "ordinary" and (
            launch_abi != "cuda_graph_gate_v1" or self.arch != core.Arch.cuda or launch is not None
        ):
            raise TaichiRuntimeError("AOT gated code requires a plain selected CUDA dispatch")
        bindings = [self.kernel_binding(arg) for arg in args]
        if self.source_arrays is not None:
            self.source_arrays.validate_kernel(args, bindings)
        if any(arg.tag not in (core.ArgKind.SCALAR, core.ArgKind.NDARRAY, core.ArgKind.TEXTURE, core.ArgKind.RWTEXTURE, core.ArgKind.ACCELERATION_STRUCTURE) for arg in args):
            raise TaichiRuntimeError("AOT execution Graph requires scalar, ndarray or qualified image/AS bindings")
        identity = (id(kernel), launch_abi)
        name = self._kernel_names.get(identity)
        if name is None:
            name = self._name("kernel" if launch_abi == "ordinary" else "gated_kernel")
            self._kernel_names[identity] = name
            self.kernels[name] = kernel
            if launch_abi != "ordinary":
                self.kernel_launch_abis[name] = launch_abi
        self._used_kernel_entries.add(name)
        operation = dict(kind="kernel", entry=name, arguments=bindings)
        if label:
            operation["label"] = label
        if launch is not None:
            if set(launch) == {"kind", "packet"} and launch["kind"] == "vulkan_indirect":
                operation["launch"] = dict(kind="vulkan_indirect", packet=self.binding(launch["packet"]))
            elif set(launch) == {"kind", "extent", "capacity"} and launch["kind"] in (
                "cpu_exact",
                "cuda_logical_exact",
            ):
                operation["launch"] = dict(launch, extent=self.binding(launch["extent"]))
            elif (
                set(launch) == {"kind", "extent", "capacity", "block_dim", "update_policy"}
                and launch["kind"] == "cuda_adaptive"
            ):
                operation["launch"] = dict(launch, extent=self.binding(launch["extent"]))
            else:
                raise TaichiRuntimeError("AOT Graph has an unsupported selected kernel launch")
        self.operations.append(operation)

    def bound_graph(self, graph, bindings):
        """Export a retained sequential graph in its own symbolic binding scope."""
        from ._graph_constraints import export_constraints

        with graph._lifecycle_lock:
            graph._check_runtime_valid()
            spec = graph._spec
            if (
                graph._workspace_lane_capacity != 1
                or graph._qualified_fusion_selector is not None
                or getattr(spec, "_storage_plans", ())
                or getattr(spec, "_binding_executor_factory", None) is not None
                or spec.control_binding_plan.names
            ):
                raise TaichiRuntimeError("AOT retained subgraph requires its fixed sequential binding recipe")
            arguments, _ = graph._snapshot_binding_source(bindings)
            if set(arguments) != set(spec.runtime_arg_names):
                raise TaichiRuntimeError("AOT retained subgraph bindings must cover its exact inputs")
            keys = ("private_bindings", "_private_names", "_scalar_values", "_scalar_symbols")
            outer = {key: getattr(self, key) for key in keys}
            outer_derived = self.reconstructed_derived_bindings
            try:
                self.reconstructed_derived_bindings = set()
                for key in keys:
                    setattr(self, key, {})
                for name, owner in arguments.items():
                    if type(owner) in (bool, int, float):
                        self.private_bindings[name] = self._scalar_values[name] = owner
                    else:
                        self.retained_arrays.bind_private(name, owner)
                self.declare_private_bindings(spec.fixed_runtime_args)
                for lease in spec.lifetime_leases:
                    exporter = type(lease).__dict__.get("_export_aot_bindings")
                    if exporter is not None:
                        exporter(lease, self)
                for node in spec.nodes:
                    _node(self, node)
                if any(
                    (
                        callable(getattr(lease, "bind_graph_arguments", None))
                        and id(lease) not in self.reconstructed_argument_binders
                    )
                    or (
                        callable(getattr(lease, "validate_graph_bindings", None))
                        and id(lease) not in self.reconstructed_binding_validators
                    )
                    for lease in spec.lifetime_leases
                ):
                    raise TaichiRuntimeError("AOT retained subgraph has an unreconstructed binding validator")
                self.binding_constraints.extend(export_constraints(spec, self.binding))
            finally:
                self.reconstructed_derived_bindings = outer_derived
                for key, value in outer.items():
                    setattr(self, key, value)


def _native(context, executable):
    action = executable.recordable_action
    if action is None:
        # Explicit concrete-type opt-in, like recording exporters below. Do not
        # inherit a base recipe through an executable with additional behavior.
        exporter = type(executable).__dict__.get("_export_aot")
        if exporter is not None:
            before = len(context.operations)
            exporter(executable, context)
            if len(context.operations) == before:
                raise TaichiRuntimeError("AOT executable export must contain an operation")
            return {}
    if action is None or action.backend_command_recording is None:
        raise TaichiRuntimeError(f"AOT native Graph needs a reconstruction description: {type(executable).__name__}")
    if action.temporary_bindings:
        raise TaichiRuntimeError("AOT native Graph temporary bindings require a resource recipe")
    if any(name in context._scalar_values for name in action.fixed_bindings):
        raise TaichiRuntimeError("AOT native fixed scalars require a typed reconstruction recipe")
    recording = action.backend_command_recording
    # Subclasses must explicitly qualify their own semantics. Inheriting the
    # base export of a callback/epilogue variant would silently lose work.
    exporter = type(recording).__dict__.get("_export_aot")
    if exporter is None:
        raise TaichiRuntimeError(f"AOT reconstruction is unavailable for recording {type(recording).__name__}")
    before = len(context.operations)
    recreated = set(exporter(recording, context))
    if len(context.operations) == before:
        raise TaichiRuntimeError("AOT recording export must contain an operation")
    if recreated | context.declared_bindings(action.fixed_bindings).keys() != set(action.fixed_bindings):
        raise TaichiRuntimeError("AOT recording did not reconstruct all private bindings")
    from taichi_forge.hardware._native_adapter import HardwareRecordingExecutable

    validate = getattr(executable, "validate_graph_bindings", None)
    if (
        getattr(validate, "__func__", None) is HardwareRecordingExecutable.validate_graph_bindings
        and executable._recording is recording
    ):
        # The standard wrapper only forwards to the exact recording whose
        # concrete export contract just succeeded. Additional validation from
        # a custom executable or an unrelated lease still needs its own recipe.
        context.reconstructed_binding_validators.add(id(executable))
    return dict(action.fixed_bindings)


def _validate_sequence(context, sequence):
    if (
        sequence._structured_depth
        or sequence._temporary_actions
        or set(sequence._derived_runtime_arg_names) - context.reconstructed_derived_bindings
    ):
        raise TaichiRuntimeError("AOT sequence requires explicit control/private resource reconstruction")
    context.require_private_bindings(sequence._fixed_runtime_args)
    for item in sequence._items:
        if item[0] not in ("dispatch", "indirect"):
            raise TaichiRuntimeError(f"AOT sequence cannot reconstruct {item[0]!r} yet")


def _cuda_nested_while(context, node):
    from taichi_forge.graph._graph import (
        _CUDA_NESTED_CONDITIONAL_ROUTE,
        _CUDA_NESTED_DEVICE_UPDATE_ROUTE,
        _CUDA_NESTED_MASKED_ROUTE,
    )

    selected = node._cuda_nested
    device_update = node._cuda_nested_control_lowering == _CUDA_NESTED_DEVICE_UPDATE_ROUTE
    masked = node._cuda_nested_control_lowering == _CUDA_NESTED_MASKED_ROUTE
    expanded = device_update or masked
    if (
        not node.supports_native_submission
        or node._cuda_nested_control_lowering
        not in (_CUDA_NESTED_CONDITIONAL_ROUTE, _CUDA_NESTED_DEVICE_UPDATE_ROUTE, _CUDA_NESTED_MASKED_ROUTE)
        or selected is None
        or not (selected.nested_expanded_eligible if expanded else selected.nested_conditional_eligible)
    ):
        raise TaichiRuntimeError("AOT Graph requires its selected nested CUDA control recipe")
    route = "masked" if masked else "device_update"
    if expanded and node._cuda_nested_submission_options.get("execution_route") != route:
        # Ordinary auto submission can fall back to masked kernels. A candidate
        # route (or one successful replay) is not a frozen physical contract.
        raise TaichiRuntimeError(f"AOT nested CUDA export requires a frozen {route.replace('_', '-')} recipe")
    if selected.physical_dispatch_count != selected.dispatch_count:
        raise TaichiRuntimeError("AOT nested CUDA control requires its selected dispatch boundaries")
    # The native nested builder owns one flat copy of the selected dispatches.
    # Its boundary table includes initial conditions and body+tail conditions;
    # exporting logical child nodes could discard physical IR or run tests twice.
    offsets = []
    first_body, boundaries = node._cuda_nested_boundaries
    operations = context.region(
        _node, selected, dispatch_offsets=offsets, gated_body_begin=first_body if masked else None
    )
    inners = node._cuda_nested_inner
    if (
        len(offsets) != selected.physical_dispatch_count + 1
        or offsets[0] != 0
        or offsets[-1] != len(operations)
        or any(left >= right for left, right in zip(offsets, offsets[1:]))
        or any(op["kind"] not in ("kernel", "native") for op in operations)
        or not 0 < first_body < selected.physical_dispatch_count
        or len(inners) != len(boundaries)
        or any(not 0 <= index <= selected.physical_dispatch_count for row in boundaries for index in row)
    ):
        raise TaichiRuntimeError("AOT nested CUDA control lacks complete physical boundaries")
    # One opaque provider dispatch can expand into several portable operations.
    # Translate selected physical boundaries, keeping the entire recording in
    # its original condition/body region rather than slicing by operation count.
    first_body = offsets[first_body]
    boundaries = tuple(tuple(offsets[index] for index in row) for row in boundaries)
    if expanded:
        if any(op["kind"] != "kernel" or "launch" in op for op in operations):
            raise TaichiRuntimeError("AOT expanded CUDA control requires plain physical kernel dispatches")
        lowering = dict(kind="cuda_nested_" + route, version=1, inners=[])
        cursor = first_body
        for inner, (condition_begin, body_begin, end) in zip(inners, boundaries):
            if not cursor <= condition_begin < body_begin < end < len(operations):
                raise TaichiRuntimeError("AOT nested CUDA control has invalid physical boundaries")
            lowering["inners"].append(
                dict(
                    predicate=context.binding(inner.predicate),
                    condition_begin=condition_begin,
                    body_begin=body_begin,
                    end=end,
                    max_iterations=inner.max_iterations,
                )
            )
            cursor = end
        context.operations.append(
            dict(
                kind="while",
                predicate=context.binding(node.predicate),
                max_iterations=node.max_iterations,
                condition=operations[:first_body],
                body=operations[first_body:],
                lowering=lowering,
            )
        )
        return
    body = []
    cursor = first_body
    for inner, (condition_begin, body_begin, end) in zip(inners, boundaries):
        if not cursor <= condition_begin < body_begin < end <= len(operations):
            raise TaichiRuntimeError("AOT nested CUDA control has invalid physical boundaries")
        body.extend(operations[cursor:body_begin])
        body.append(
            dict(
                kind="while",
                predicate=context.binding(inner.predicate),
                max_iterations=inner.max_iterations,
                condition=[],
                body=operations[body_begin:end],
            )
        )
        cursor = end
    body.extend(operations[cursor:])
    context.operations.extend(operations[:first_body])
    context.operations.append(
        dict(
            kind="while",
            predicate=context.binding(node.predicate),
            max_iterations=node.max_iterations,
            condition=[],
            body=body,
        )
    )


def _vulkan_nested_while(context, node):
    selected = node._vulkan_nested
    if not node.supports_native_submission or selected is None:
        raise TaichiRuntimeError("AOT Graph requires its selected nested Vulkan control recipe")
    if selected.physical_dispatch_count != selected.dispatch_count:
        raise TaichiRuntimeError("AOT nested Vulkan control requires its selected dispatch boundaries")
    operations = context.region(_node, selected)
    initial, boundaries = node._vulkan_nested_boundaries
    inners = node._vulkan_nested_inner
    if (
        len(operations) != selected.physical_dispatch_count
        or any(op["kind"] != "kernel" for op in operations)
        or not 0 < initial < len(operations)
        or len(inners) != len(boundaries)
    ):
        raise TaichiRuntimeError("AOT nested Vulkan control lacks complete physical boundaries")
    lowering = dict(kind="vulkan_nested", version=1, counter=context.binding(node.counter), inners=[])
    if node.status is not None:
        lowering["status"] = context.binding(node.status)
    cursor = initial
    for inner, (condition_begin, body_begin, end), budget, chunk in zip(
        inners, boundaries, node._nested_inner_max_iterations, node._nested_inner_compound_chunk_limits, strict=True
    ):
        if not cursor <= condition_begin < body_begin < end < len(operations):
            raise TaichiRuntimeError("AOT nested Vulkan control has invalid physical boundaries")
        value = dict(
            predicate=context.binding(inner.predicate),
            counter=context.binding(inner.counter),
            condition_begin=condition_begin,
            body_begin=body_begin,
            end=end,
            max_iterations=budget,
            chunk_size=chunk,
        )
        if inner.status is not None:
            value["status"] = context.binding(inner.status)
        lowering["inners"].append(value)
        cursor = end
    # All tail conditions are already present in the selected physical body.
    # Keep its initial condition inside the same native recording, once only.
    context.operations.append(
        dict(
            kind="while",
            predicate=context.binding(node.predicate),
            max_iterations=node.max_iterations,
            condition=operations[:initial],
            body=operations[initial:],
            lowering=lowering,
        )
    )


def _while(context, node):
    from taichi_forge.lang import impl

    arch = impl.current_cfg().arch
    if arch == core.Arch.vulkan:
        from taichi_forge.graph._graph import _vulkan_compound_strategy_codes

        if not node.max_iterations:
            _node(context, node._condition)
            return
        if node._has_nested_control:
            _vulkan_nested_while(context, node)
            return
        if not node.supports_native_submission:
            raise TaichiRuntimeError("AOT Graph requires its selected Vulkan control recipe")
        selected = node._vulkan_structured
        if selected is None or selected.physical_dispatch_count != selected.dispatch_count:
            raise TaichiRuntimeError("AOT Vulkan control requires its selected dispatch boundaries")
        operations = context.region(_node, selected)
        boundary = node.condition_dispatch_count
        if not 0 < boundary < len(operations) or any(op["kind"] != "kernel" for op in operations):
            raise TaichiRuntimeError("AOT Vulkan control requires flat selected kernel regions")
        # Freeze the same strategy selection used by native submission. The
        # consumer never reads producer environment variables or reselects it.
        strategies = _vulkan_compound_strategy_codes(node.compound_chunk_count, node.vulkan_first_chunk_strategy)
        names = ("compact", "compact", "chained", "conditional", "coarse_conditional")
        lowering = dict(
            kind="vulkan_structured",
            version=1,
            counter=context.binding(node.counter),
            chunks=[
                dict(iterations=count, strategy=names[strategy])
                for count, strategy in zip(node._vulkan_chunk_iterations, strategies, strict=True)
            ],
        )
        if node.status is not None:
            lowering["status"] = context.binding(node.status)
        context.operations.extend(operations[:boundary])
        context.operations.append(
            dict(
                kind="while",
                predicate=context.binding(node.predicate),
                max_iterations=node.max_iterations,
                condition=[],
                body=operations[boundary:],
                lowering=lowering,
            )
        )
        return
    if arch == core.Arch.cuda:
        if node._has_nested_control:
            _cuda_nested_while(context, node)
            return
        if node.supports_native_submission and node._cuda_control_lowering == "cuda_masked_bounded_graph":
            _node(context, node._condition)
            if node.max_iterations:
                # The selected chunk already includes its tail condition. Gate
                # that complete physical body; zero budgets export no body code.
                body = context.region(_node, node._chunks[1], gated_body_begin=0)
                context.operations.append(
                    dict(
                        kind="while",
                        predicate=context.binding(node.predicate),
                        max_iterations=node.max_iterations,
                        condition=[],
                        body=body,
                        lowering=dict(kind="cuda_masked", version=1),
                    )
                )
            return
        if not node.supports_native_submission or node._cuda_control_lowering != "cuda_conditional_graph":
            raise TaichiRuntimeError("AOT Graph requires its selected CUDA control recipe")
    elif arch not in (core.Arch.x64, core.Arch.arm64):
        raise TaichiRuntimeError("AOT Graph requires its selected backend control recipe")
    elif node.chunk_limit != 1:
        raise TaichiRuntimeError("AOT Graph requires its selected chunked CPU control recipe")
    # Preserve the physical body+condition segment. Splitting or rebuilding it
    # could discard selected fusion; appending condition again repeats work.
    _node(context, node._condition)
    if node.max_iterations:
        context.operations.append(
            dict(
                kind="while",
                predicate=context.binding(node.predicate),
                max_iterations=node.max_iterations,
                condition=[],
                body=context.region(_node, node._chunks[1]),
            )
        )


def _parallel(context, node):
    """Preserve the selected physical fork/join and its local dispatch order."""
    if context.arch != core.Arch.cuda:
        raise TaichiRuntimeError("AOT parallel requires its selected CUDA backend")
    sources, native_groups = node.compiled_graph._physical_jit_parallel_execution_sources
    groups = tuple(tuple(group) for group in native_groups)
    if groups != node.parallel_dispatch_groups or len(groups) < 2 or not all(groups):
        raise TaichiRuntimeError("AOT parallel metadata does not match the materialized Graph")
    indices = tuple(index for group in groups for index in group)
    if (
        len(sources) != node.physical_dispatch_count
        or indices != tuple(range(indices[0], indices[-1] + 1))
        or indices[0] < 0
        or indices[-1] >= len(sources) - 1
        or any(source[0] != "dispatch" or source[-1] is not None for source in sources)
    ):
        raise TaichiRuntimeError("AOT parallel requires contiguous ordinary branches and a join suffix")

    def emit(context, indices):
        for index in indices:
            context.kernel(*sources[index][1:])

    emit(context, range(indices[0]))
    context.operations.append(dict(kind="cuda_parallel", branches=[context.region(emit, group) for group in groups]))
    emit(context, range(indices[-1] + 1, len(sources)))


def _branch(context, node):
    from taichi_forge.graph._graph import _CompiledIfGraphNode

    is_if = isinstance(node, _CompiledIfGraphNode)
    selector = node.predicate if is_if else node.selector
    fallback = node._else if is_if else node._default
    children = (node._then,) if is_if else node._branches
    if fallback is not None:
        children = (*children, fallback)
    if context.arch == core.Arch.vulkan or (
        context.arch == core.Arch.cuda and node.lowering_mode == "portable"
    ):
        # These routes already select on the host in JIT. Preserve that choice;
        # a CUDA auto route remains ineligible because it may upgrade at replay.
        bodies = [context.region(_node, child) for child in children]
        lowering = "vulkan_host" if context.arch == core.Arch.vulkan else "cuda_host"
    elif context.arch == core.Arch.cuda:
        selected = node._native_branches
        masked = node._cuda_control_lowering == "cuda_masked_bounded_graph"
        if (
            not node.supports_native_submission
            or node._cuda_control_lowering not in ("cuda_conditional_graph", "cuda_masked_bounded_graph")
            or selected is None
        ):
            raise TaichiRuntimeError("AOT Graph requires its selected CUDA branch recipe")
        counts = node._native_branch_dispatch_counts
        if (
            len(counts) != len(children)
            or any(count <= 0 for count in counts)
            or sum(counts) != selected.physical_dispatch_count
            or (not masked and not selected.compiled_graph._cuda_conditional_body_eligible())
        ):
            raise TaichiRuntimeError("AOT CUDA branch requires its selected qualified conditional-body boundaries")
        # Export the actual combined physical sequence used by the native branch
        # builder. Recompiling individual logical children could lose its recipe.
        offsets = []
        operations = context.region(_node, selected, dispatch_offsets=offsets, gated_body_begin=0 if masked else None)
        if len(offsets) != sum(counts) + 1 or offsets[-1] != len(operations):
            raise TaichiRuntimeError("AOT CUDA branch lacks complete physical dispatch metadata")
        bodies, cursor = [], 0
        for count in counts:
            bodies.append(operations[offsets[cursor] : offsets[cursor + count]])
            cursor += count
        lowering = "cuda_masked" if masked else "cuda_conditional"
    elif context.arch in (core.Arch.x64, core.Arch.arm64):
        bodies = [context.region(_node, child) for child in children]
        lowering = "cpu"
    else:
        raise TaichiRuntimeError("AOT Graph requires its selected backend branch recipe")
    condition = context.region(_node, node._condition)
    if lowering == "cuda_masked" and any(op["kind"] != "kernel" or "launch" in op for op in condition):
        raise TaichiRuntimeError("AOT masked branch requires a flat kernel condition without launch overrides")
    context.operations.append(
        dict(
            kind="branch",
            selector=context.binding(selector),
            selection="nonzero" if is_if else "indexed",
            has_default=fallback is not None,
            condition=condition,
            bodies=bodies,
            lowering=dict(kind=lowering, version=1),
        )
    )


def _node(context, node, *, dispatch_offsets=None, gated_body_begin=None):
    from taichi_forge.graph._graph import (
        _CompiledCGraphNode,
        _CompiledIfGraphNode,
        _CompiledNativeGraphNode,
        _CompiledSequentialRegionNode,
        _CompiledSwitchGraphNode,
        _CompiledWhileGraphNode,
    )
    from taichi_forge.graph._ir import DispatchNode, SequentialRegion

    if (dispatch_offsets is not None or gated_body_begin is not None) and not isinstance(node, _CompiledCGraphNode):
        raise TaichiRuntimeError("AOT dispatch boundaries require a flat physical Graph")
    if isinstance(node, _CompiledSequentialRegionNode):
        for child in node.nodes:
            _node(context, child)
        return
    if isinstance(node, _CompiledWhileGraphNode):
        _while(context, node)
        return
    if isinstance(node, (_CompiledIfGraphNode, _CompiledSwitchGraphNode)):
        _branch(context, node)
        return
    if isinstance(node, _CompiledNativeGraphNode):
        if set(node.derived_runtime_arg_names) - context.reconstructed_derived_bindings:
            raise TaichiRuntimeError("AOT native derived bindings require a reconstruction recipe")
        _native(context, node.executable)
        return
    if not isinstance(node, _CompiledCGraphNode):
        raise TaichiRuntimeError(f"AOT execution Graph cannot reconstruct {type(node).__name__} yet")
    context.field_state(node.snode_tree_dependency_info)
    if node.temporary_actions or set(node.derived_runtime_arg_names) - context.reconstructed_derived_bindings:
        raise TaichiRuntimeError("AOT execution Graph requires the selected composition/resource recipe")
    ir_nodes = node.ir_node.children if isinstance(node.ir_node, SequentialRegion) else (node.ir_node,)
    from ._bounded_export import validate_domains

    validate_domains(context, ir_nodes, node.recording_dispatches)
    if any(
        manifest.execution_kind not in ("kernel_dispatch", "indirect_dispatch", "backend_command")
        for manifest in node.native_action_manifests
    ):
        raise TaichiRuntimeError("AOT execution Graph contains an unsupported native execution kind")
    command_count = sum(manifest.execution_kind == "backend_command" for manifest in node.native_action_manifests)
    if gated_body_begin is not None and (
        command_count or node.parallel_dispatch_groups or not 0 <= gated_body_begin < node.physical_dispatch_count
    ):
        raise TaichiRuntimeError("AOT masked control requires a flat physical kernel body")
    if not command_count:
        context.require_private_bindings(node.fixed_runtime_args)
        if len(node.recording_dispatches) != node.dispatch_count:
            raise TaichiRuntimeError("AOT execution Graph lacks complete symbolic dispatch metadata")
        if node.parallel_dispatch_groups:
            if dispatch_offsets is not None:
                raise TaichiRuntimeError("AOT nested control needs explicit parallel-region boundaries")
            _parallel(context, node)
            return
        dispatches = node.compiled_graph._physical_jit_execution_sources
        if len(dispatches) != node.physical_dispatch_count:
            raise TaichiRuntimeError("AOT execution Graph lacks complete physical dispatch metadata")
        for index, dispatch in enumerate(dispatches):
            if dispatch[0] != "dispatch":
                raise TaichiRuntimeError("AOT kernel sequence contains an unexpected native command")
            if dispatch_offsets is not None:
                dispatch_offsets.append(len(context.operations))
            context.kernel(
                *dispatch[1:],
                launch_abi=(
                    "cuda_graph_gate_v1" if gated_body_begin is not None and index >= gated_body_begin else "ordinary"
                ),
            )
        if dispatch_offsets is not None:
            dispatch_offsets.append(len(context.operations))
        return
    # Native commands are composition barriers. Preserve their retained opaque
    # descriptions, but export kernels from the final physical sequence: source
    # maps may have been replaced by fused IR during ordinary compilation too.
    natives, sequences = [], []
    kernel_count = 0
    for item in node.recording_operations:
        if item[0] == "native":
            natives.append(item[1])
        elif item[0] in ("dispatch", "indirect"):
            kernel_count += 1
        elif item[0] == "sequential":
            sequences.append(item[1])
            kernel_count += len(item[1]._items)
        else:
            raise TaichiRuntimeError(f"AOT execution Graph cannot reconstruct recipe {item[0]!r}")
    if (
        len(natives) != command_count
        or kernel_count != len(node.recording_dispatches)
        or len(natives) + kernel_count != node.dispatch_count
    ):
        raise TaichiRuntimeError("AOT recording sequence does not cover the materialized Graph")
    sources = node.compiled_graph._physical_jit_execution_sources
    if len(sources) != node.physical_dispatch_count:
        raise TaichiRuntimeError("AOT mixed Graph lacks complete physical recording metadata")
    recreated = context.declared_bindings(node.fixed_runtime_args)
    native_count = 0
    for source in sources:
        if dispatch_offsets is not None:
            dispatch_offsets.append(len(context.operations))
        if source[0] == "dispatch":
            context.kernel(*source[1:])
        elif source[0] == "native" and source[1] == native_count and native_count < len(natives):
            for name, owner in _native(context, natives[native_count]).items():
                if name in recreated and recreated[name] is not owner:
                    raise TaichiRuntimeError("AOT recording private binding owners conflict")
                recreated[name] = owner
            native_count += 1
        else:
            raise TaichiRuntimeError("AOT mixed Graph has an invalid physical recording source")
    if dispatch_offsets is not None:
        dispatch_offsets.append(len(context.operations))
    if native_count != len(natives):
        raise TaichiRuntimeError("AOT mixed Graph lacks complete native recording coverage")
    for sequence in sequences:
        _validate_sequence(context, sequence)
    if recreated.keys() != node.fixed_runtime_args.keys() or any(
        recreated[name] is not value for name, value in node.fixed_runtime_args.items()
    ):
        raise TaichiRuntimeError("AOT execution Graph has unreconstructed fixed bindings")


def export_graph(module, graph, *, gradients=None, snapshot_state=False, source_bindings=None, resource_bindings=None):
    """Preflight the entire region before publishing any module definitions."""
    with graph._lifecycle_lock:
        graph._check_runtime_valid()
        if core.arch_name(module._arch) != graph._execution_arch:
            raise TaichiRuntimeError("AOT execution Graph and module backends must match")
        if graph._workspace_lane_capacity != 1 or graph._qualified_fusion_selector is not None:
            raise TaichiRuntimeError("AOT execution Graph needs a fixed single-frame execution recipe")
        spec = graph._spec
        if getattr(spec, "_storage_plans", ()):
            raise TaichiRuntimeError("AOT execution Graph requires the selected storage/binding recipe")
        binding_factory = getattr(spec, "_binding_executor_factory", None)
        binding_strategy = None
        if binding_factory is not None:
            # Only a concrete executor can qualify equivalence with the fixed
            # AOT submission path. Inherited hooks could lose queue/segment work.
            if getattr(binding_factory, "__dict__", {}).get("_validate_aot_export") is None:
                raise TaichiRuntimeError("AOT execution Graph requires the selected storage/binding recipe")
            binding_strategy = binding_factory._validate_aot_export(spec, graph._execution_arch)
        gradient_names = set((gradients or {}).values())
        from ._graph_source_bindings import SourceArrayBindings
        from ._graph_resource_sources import SourceResourceBindings
        from ._graph_resource_bindings import ResourceInputBindings

        source_resources = SourceResourceBindings(source_bindings, spec.fixed_runtime_args)
        source_arrays = SourceArrayBindings(source_resources.arrays, spec.fixed_runtime_args)
        resource_inputs = ResourceInputBindings(resource_bindings, spec.fixed_runtime_args)
        non_resource_names = (
            (set(spec.runtime_arg_names) - source_resources.owners.keys())
            | source_arrays.arrays.keys()
            | gradient_names
        )
        if resource_inputs.public_names & non_resource_names:
            raise ValueError("Public resource names conflict with Graph array/scalar/gradient arguments")
        context = _ExportContext(
            module,
            set(spec.runtime_arg_names)
            | set(spec.fixed_runtime_args)
            | gradient_names
            | source_arrays.arrays.keys()
            | source_resources.owners.keys()
            | resource_inputs.names,
            snapshot_state=snapshot_state,
            source_arrays=source_arrays,
            source_resources=source_resources,
            resource_inputs=resource_inputs,
        )
        context.declare_private_bindings(spec.fixed_runtime_args)
        source_resources.images.preflight(context.arch, spec._texture_binding_requirements)
        context.control_names = set(spec.control_binding_plan.names)
        # Dispatch actions may already be flattened into ordinary physical
        # kernels. Reconstruct their concrete binding contracts before lowering
        # those kernels, without regenerating the selected dispatch recipe.
        for lease in spec.lifetime_leases:
            exporter = type(lease).__dict__.get("_export_aot_bindings")
            if exporter is not None:
                exporter(lease, context)
        for node in graph._spec.nodes:
            _node(context, node)
        source_arrays.validate_used(spec.runtime_arg_names)
        source_resources.validate_used()
        resource_inputs.validate_used()
        context.resolve_resource_aliases()
        context.binding_constraints.extend(source_arrays.constraints)
        if context.control_names != context._control_bindings:
            raise TaichiRuntimeError("AOT Graph requires complete selected control binding metadata")
        for _, bindings in spec.control_binding_plan.distinct_groups:
            names = [context.binding(name) for _, _, name in bindings]
            for i, left in enumerate(names):
                for right in names[i + 1 :]:
                    context.binding_constraints.append(dict(kind="disjoint", left=left, right=right))
        if any(
            (
                callable(getattr(lease, "bind_graph_arguments", None))
                and id(lease) not in context.reconstructed_argument_binders
            )
            or (
                callable(getattr(lease, "validate_graph_bindings", None))
                and id(lease) not in context.reconstructed_binding_validators
            )
            for lease in spec.lifetime_leases
        ):
            raise TaichiRuntimeError("AOT execution Graph requires the selected binding validation recipe")
        mapped_gradients = {}
        for primal, gradient in (gradients or {}).items():
            if primal in context._scalar_values or gradient in context._scalar_values:
                raise TaichiRuntimeError("AOT gradient bindings require ndarrays")
            primal, gradient = context.binding(primal), context.binding(gradient)
            if primal in mapped_gradients and mapped_gradients[primal] != gradient:
                raise TaichiRuntimeError("AOT aliases of one primal must share the same gradient resource")
            mapped_gradients[primal] = gradient
        from ._graph_constraints import export_constraints

        constraints = export_constraints(spec, context.binding, additional=context.binding_constraints)
        try:
            context.retained_arrays.publish()
            for name, kernel in context.kernels.items():
                if name in context.kernel_launch_abis:
                    module._aot_builder._add_with_launch_abi(name, kernel, context.kernel_launch_abis[name])
                else:
                    module._aot_builder.add(name, kernel)
            # A different AOT compile request can retain a different Field
            # closure than the selected JIT graph. Inspect the actual exports,
            # then sample roots/provider arrays at one completion boundary.
            context.materialize_provider_data(module)
            plan = ExecutionPlan(
                context.operations,
                initialization=context.initialization or None,
                resources=context.resources,
                gradients=mapped_gradients or None,
                constraints=constraints or None,
                binding_strategy=binding_strategy,
            )
            # Provider workspace must not accidentally become a public input.
            expected_names = (
                (set(spec.runtime_arg_names) - source_resources.used)
                | source_arrays.used
                | resource_inputs.used_names
                | (gradient_names - context.private_bindings.keys())
            )
            if set(plan.argument_names) != expected_names:
                raise TaichiRuntimeError("AOT Graph reconstruction changed the public binding contract")
            for key in plan.data_entries:
                if key not in context.data_payloads:
                    raise TaichiRuntimeError("AOT Graph reconstruction omitted an initial data payload")
                if key in module._data_payloads and module._data_payloads[key] != context.data_payloads[key]:
                    raise ValueError("AOT data digest collision")
            module._add_initial_root_snapshots(context.root_dependencies, context.root_payloads)
        except Exception:
            # Builders cannot roll back compiled entries. Finalization failures
            # also forbid publication, even if earlier kernels compiled.
            module._graph_export_failed = True
            raise
        module._exported_kernels.update(context.kernels)
        module._kernel_launch_abis.update(context.kernel_launch_abis)
        module._content.extend("kernel:" + name for name in context.kernels)
        module._data_payloads.update({key: context.data_payloads[key] for key in plan.data_entries})
        return plan
