"""Persist the binding contract of an already selected Graph CG/PCG solve."""

from taichi_forge.lang.exception import TaichiRuntimeError


def export_bindings(executable, context):
    executable.validate_graph_lifetime()
    if executable._update_policy != "pinned":
        raise TaichiRuntimeError("AOT SolvePlan requires graph_action(update_policy='pinned')")
    rhs, output, initial = executable._rhs_name, executable._output_name, executable._initial_name
    extents = {rhs: executable._operator_shape[0], output: executable._operator_shape[0]}
    if initial is not None:
        extents[initial] = executable._operator_shape[0]
    extents.update({executable._terminal.state.name: 4, executable._terminal.metrics.name: 4})
    for name, extent in extents.items():
        context.binding_constraints.append(
            dict(
                kind="array_layout",
                array=context.binding(name),
                exact_shape=[extent],
                element_shape=[],
                record_bytes=4,
                alignment=4,
            )
        )
    pairs = [(rhs, output, False)]
    if initial not in (None, output):
        pairs.extend(((rhs, initial, False), (initial, output, True)))
    for left, right, same_view in pairs:
        requirement = dict(kind="disjoint", left=context.binding(left), right=context.binding(right))
        if same_view:
            requirement["allow_same_view"] = True
        context.binding_constraints.append(requirement)
    # AOT ports are typed, packed ndarrays (including native views). Their
    # layout checks reconstruct the no-staging vector binder in the cold phase.
    # Operator/preconditioner leases export their own exact pinned generations;
    # the shared Graph exporter retains all private scratch and selected control.
    context.reconstructed_argument_binders.add(id(executable))
    context.reconstructed_binding_validators.add(id(executable))
