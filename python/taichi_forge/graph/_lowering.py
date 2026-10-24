"""Cold lowering of an already selected Graph into an ExecutionPlan description.

The common layer owns traversal, physical operation boundaries, symbolic names
and binding constraints. Adapters supply code references and storage policies;
publication, snapshots, code loans and runtime ownership stay outside this file.
"""

import ctypes
import json
import struct

from taichi_forge._lib import core
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge_runtime.aot import Scalar


def _snapshot(value):
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


class GraphLoweringContext:
    """Shared cold construction state, without a module or export transaction.

    ``_kernel_entry`` and ``field_state`` are ownership adapters. The retained
    array adapter handles concrete storage references; optional data recipes
    remain methods of the corresponding AOT or live-loan adapter.
    """

    def __init__(self, arch, reserved_names, *, source_arrays, source_resources, resource_inputs):
        self.arch = arch
        self.source_arrays = source_arrays
        self.source_resources = source_resources
        self.source_images = source_resources.images if source_resources is not None else None
        self.resource_inputs = resource_inputs
        self.operations = []
        self.initialization = []
        self.resources = []
        self.storage_recipe = None
        self._temporary_storage_groups = {}
        self._initialization_resources = {}
        self._used_kernel_entries = set()
        self._buffer_only_kernels = True
        self._state_dependencies = {}
        self._resource_owners = {}
        self._resource_policies = {}
        self._public_resource_owners = {}
        self._resource_links = {}
        self._indices = {}
        self._names = set(reserved_names)
        self._kernel_names = {}
        self.reconstructed_binding_validators = set()
        self.reconstructed_argument_binders = set()
        self.reconstructed_derived_bindings = set()
        self.reconstructed_temporary_actions = set()
        self.binding_constraints = []
        self._native_export_facts = {}
        self.bounded_extents = {}
        self.private_bindings = {}
        self._private_names = {}
        self._private_allocators = {}
        self._scalar_values = {}
        self._scalar_symbols = {}
        self._scalar_resources = {}
        self.control_names = set()
        self._control_bindings = set()

    def _capture_private_array(self, owner):
        return False

    def _kernel_entry(self, kernel, launch_abi, jit_source):
        raise NotImplementedError("The lowering adapter must reference selected kernel code")

    def field_state(self, dependencies):
        raise NotImplementedError("The lowering adapter must retain Field dependencies")

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

    def immutable_resource_inputs(self, owner, assets):
        """AOT requires explicit deployable assets; a JIT adapter may retain them."""
        return self.resource_inputs.require(owner, tuple(assets), asset_ports=tuple(assets))

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

    def declare_storage_plans(self, graph):
        from taichi_forge.graph._recipes.runtime_storage import validate_storage_plans

        spec = graph._spec
        plans = getattr(spec, "_storage_plans", ())
        if not plans:
            return
        validate_storage_plans(spec, plans)
        for plan, owner in zip(plans, graph._instance._storage_owners, strict=True):
            # A concrete factory must preserve its chosen allocation policy.
            # Arbitrary callbacks and inherited hooks cannot qualify a recipe.
            exporter = type(plan.factory).__dict__.get("_export_aot_allocator")
            if exporter is None:
                raise TaichiRuntimeError("AOT execution Graph requires the selected storage/binding recipe")
            allocator = exporter(plan.factory, self, plan, owner)
            if plan.temporary_arena:
                from ._lowering_temporaries import declare_ordered_temporaries

                declare_ordered_temporaries(self, graph, allocator)
            for name in plan.binding_names:
                identity = id(graph._instance._fixed_runtime_args[name])
                previous = self._private_allocators.setdefault(identity, allocator)
                if previous != allocator:
                    raise TaichiRuntimeError("AOT private aliases have conflicting storage owners")

    def declare_private_bindings(self, bindings, instance):
        from taichi_forge.graph._ir import InternalNdarrayRequirement
        from taichi_forge.lang._storage_view import DenseNdarrayView
        from taichi_forge.lang.util import cook_dtype

        captured = []
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
            if type(owner) is DenseNdarrayView:
                # Reconstruct its finite packed range against a named parent or
                # qualified module root; never widen it to the backing storage.
                captured.append((name, owner))
                continue
            if self._capture_private_array(owner):
                captured.append((name, owner))
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
                # A frozen declaration can be shared by distinct compiled
                # instances. Only aliases of the actual instance allocation
                # share storage; repeated uses of that instance still dedup.
                instance._fixed_runtime_args[name],
                dict(
                    kind="ndarray",
                    dtype=element_type,
                    shape=owner.shape,
                    element_shape=dtype.shape(),
                    zero_initialize=True,
                    **(
                        {"allocator": self._private_allocators[id(instance._fixed_runtime_args[name])]}
                        if id(instance._fixed_runtime_args[name]) in self._private_allocators
                        else {}
                    ),
                ),
            )
            self.private_bindings[name] = owner
        return captured

    def declare_recipe_bindings(self, graph):
        """Resolve explicit reconstruction roles before generic state capture."""
        spec = graph._spec
        self.declare_storage_plans(graph)
        captured = self.declare_private_bindings(spec.fixed_runtime_args, graph._instance)
        for lease in spec.lifetime_leases:
            exporter = type(lease).__dict__.get("_export_aot_bindings")
            if exporter is not None:
                exporter(lease, self)
        # A qualified producer may already have proved scratch/data ownership.
        # Reuse that resource; only otherwise-unclassified arrays become state.
        for name, owner in captured:
            self.retained_arrays.bind_private(name, owner)

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

    def kernel_binding(self, arg):
        if arg.tag in (
            core.ArgKind.TEXTURE,
            core.ArgKind.RWTEXTURE,
            core.ArgKind.TEXTURE_COLLECTION,
            core.ArgKind.ACCELERATION_STRUCTURE,
        ):
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

    def kernel(self, kernel, args, label="", launch=None, *, launch_abi="ordinary", jit_source=None):
        if launch_abi != "ordinary" and (
            launch_abi != "cuda_graph_gate_v1" or self.arch != core.Arch.cuda or launch is not None
        ):
            raise TaichiRuntimeError("AOT gated code requires a plain selected CUDA dispatch")
        has_packs = any(arg.tag == core.ArgKind.ARGPACK for arg in args)
        name = self._kernel_entry(kernel, launch_abi, jit_source)
        if has_packs:
            from ._lowering_argpacks import kernel_pack_bindings

            bindings = kernel_pack_bindings(self, args, name, jit_source)
        else:
            bindings = [self.kernel_binding(arg) for arg in args]
        if any(arg.tag not in (core.ArgKind.SCALAR, core.ArgKind.NDARRAY) for arg in args):
            self._buffer_only_kernels = False
        if self.source_arrays is not None and not has_packs:
            self.source_arrays.validate_kernel(args, bindings)
        if any(
            arg.tag
            not in (
                core.ArgKind.SCALAR,
                core.ArgKind.NDARRAY,
                core.ArgKind.TEXTURE,
                core.ArgKind.RWTEXTURE,
                core.ArgKind.TEXTURE_COLLECTION,
                core.ArgKind.ACCELERATION_STRUCTURE,
                core.ArgKind.ARGPACK,
            )
            for arg in args
        ):
            raise TaichiRuntimeError("AOT execution Graph requires scalar, ndarray or qualified image/AS bindings")
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
        from ._lowering_constraints import export_constraints

        with graph._lifecycle_lock:
            graph._check_runtime_valid()
            spec = graph._spec
            if (
                graph._workspace_lane_capacity != 1
                or graph._qualified_fusion_selector is not None
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
                self.declare_recipe_bindings(graph)
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
    if arch == core.Arch.cuda and node.lowering_mode == "portable":
        if node.chunk_limit != 1:
            raise TaichiRuntimeError("AOT host while requires its selected exact iteration recipe")
        # The selected body already includes its tail condition. Keep that
        # physical segment, including provider boundaries and selected fusion.
        _node(context, node._condition)
        if node.max_iterations:
            context.operations.append(
                dict(
                    kind="while",
                    predicate=context.binding(node.predicate),
                    max_iterations=node.max_iterations,
                    condition=[],
                    body=context.region(_node, node._chunks[1]),
                    lowering=dict(kind="cuda_host", version=1),
                )
            )
        return
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
            context.kernel(*sources[index][1:], jit_source=(node.compiled_graph, index))

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
    if context.arch == core.Arch.vulkan or (context.arch == core.Arch.cuda and node.lowering_mode == "portable"):
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
    if (
        any(id(action) not in context.reconstructed_temporary_actions for action in node.temporary_actions)
        or set(node.derived_runtime_arg_names) - context.reconstructed_derived_bindings
    ):
        raise TaichiRuntimeError("AOT execution Graph requires the selected composition/resource recipe")
    ir_nodes = node.ir_node.children if isinstance(node.ir_node, SequentialRegion) else (node.ir_node,)
    from ._lowering_bounded import validate_domains

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
                jit_source=(node.compiled_graph, index),
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
    for index, source in enumerate(sources):
        if dispatch_offsets is not None:
            dispatch_offsets.append(len(context.operations))
        if source[0] == "dispatch":
            context.kernel(*source[1:], jit_source=(node.compiled_graph, index))
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


def lower_frozen_graph(context, graph):
    """Lower one complete selected recipe for either an artifact or a live loan.

    Both owners require the same control boundaries and reconstructed binding
    contracts. Only code publication and live storage ownership differ.
    """
    spec = graph._spec
    context.source_resources.images.preflight(context.arch, spec._texture_binding_requirements)
    context.source_resources.argpacks.preflight(context, spec._argpack_binding_requirements)
    context.control_names = set(spec.control_binding_plan.names)
    context.declare_recipe_bindings(graph)
    operation_ends = []
    for node in spec.nodes:
        _node(context, node)
        operation_ends.append(len(context.operations))
    context.source_arrays.validate_used(spec.runtime_arg_names)
    context.source_resources.validate_used()
    context.resource_inputs.validate_used()
    context.resolve_resource_aliases()
    context.binding_constraints.extend(context.source_arrays.constraints)
    if context.control_names != context._control_bindings:
        raise TaichiRuntimeError("Graph requires complete selected control binding metadata")
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
        raise TaichiRuntimeError("Graph requires the selected binding validation recipe")
    return tuple(operation_ends)
