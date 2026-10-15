"""Cold bounded-work ABI certificates; selected launch metadata stays native."""

from taichi_forge.lang.exception import TaichiRuntimeError


def _array(context, name, shape):
    context.binding_constraints.append(
        dict(
            kind="array_layout",
            array=context.binding(name),
            exact_shape=shape,
            element_shape=[],
            record_bytes=4,
            alignment=4,
            dtype="i32",
        )
    )


def extent_contract(contract, context):
    contract.validate_graph_lifetime()
    if contract._expected_extent is not None:
        raise TaichiRuntimeError("AOT producer-owned extent requires its packet reconstruction recipe")
    name = context.binding(contract.extent_name)
    previous = context.bounded_extents.setdefault(name, contract.capacity)
    if previous != contract.capacity:
        raise TaichiRuntimeError("AOT bounded extent has conflicting capacities")
    _array(context, contract.extent_name, [2])
    context.reconstructed_binding_validators.add(id(contract))


def offsets_contract(contract, context):
    _array(context, contract.offsets_name, [contract.segment_count + 1])
    context.reconstructed_binding_validators.add(id(contract))


def validate_domains(context, nodes, dispatches):
    from taichi_forge.graph._ir import DispatchNode

    if not any(isinstance(node, DispatchNode) and node.bounded_domain is not None for node in nodes):
        return
    if len(nodes) != len(dispatches):
        raise TaichiRuntimeError("AOT bounded dispatch lacks complete symbolic launch metadata")
    for node, dispatch in zip(nodes, dispatches):
        domain = node.bounded_domain if isinstance(node, DispatchNode) else None
        if domain is None:
            continue
        if domain.count_source != "device_extent":
            raise TaichiRuntimeError("AOT host bounded counts require their clamping recipe")
        if context.bounded_extents.get(context.binding(domain.extent)) != domain.capacity:
            raise TaichiRuntimeError("AOT bounded dispatch requires its extent contract")
        if dispatch.dispatch_packet is None and domain.physical_grid_requirement not in (
            "auto", "fixed_capacity", "logical_exact", "require_exact"
        ):
            raise TaichiRuntimeError("AOT exact bounded dispatch requires its backend launch recipe")
