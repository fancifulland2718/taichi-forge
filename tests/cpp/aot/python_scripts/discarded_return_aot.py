"""CGraph side effects, discarded results and an explicit adjoint chain."""

import taichi_forge as ti


def export(directory):
    @ti.kernel
    def advance(
        state: ti.types.ndarray(ti.i32, ndim=1), amount: ti.i32
    ) -> ti.types.vector(3, ti.i32):
        state[0] += amount
        return ti.Vector([state[0], amount, -amount])

    @ti.kernel
    def publish(state: ti.types.ndarray(ti.i32, ndim=1)):
        state[1] = state[0]

    @ti.kernel
    def energy(
        x: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
        loss: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True),
    ):
        for i in range(4):
            loss[0] += x[i] * x[i]

    @ti.kernel
    def read_energy(loss: ti.types.ndarray(ti.f32, ndim=1, needs_grad=True)) -> ti.f32:
        return loss[0]

    builder = ti.graph.GraphBuilder()
    state = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "state", ti.i32, ndim=1)
    amount = ti.graph.Arg(ti.graph.ArgKind.SCALAR, "amount", ti.i32)
    x = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "x", ti.f32, ndim=1)
    loss = ti.graph.Arg(ti.graph.ArgKind.NDARRAY, "loss", ti.f32, ndim=1)
    builder.dispatch(advance, state, amount)
    builder.dispatch(advance, state, amount)
    builder.dispatch(publish, state)
    builder.dispatch(energy, x, loss)
    builder.dispatch(read_energy, loss)
    builder.dispatch(energy.grad, x, loss)
    module = ti.aot.Module()
    module.add_kernel(advance)
    module.add_kernel(read_energy)
    module.add_kernel(publish)
    module.add_kernel(energy)
    module.add_kernel(energy.grad, name="backward")
    module.add_graph("step", builder.compile())
    module.add_execution_plan("step", ti.aot.ExecutionPlan([
        dict(kind="kernel", entry="advance", arguments=["state", "amount"]),
        dict(kind="kernel", entry="advance", arguments=["state", "amount"]),
        dict(kind="kernel", entry="publish", arguments=["state"]),
        dict(kind="kernel", entry="energy", arguments=["x", "loss"]),
        dict(kind="kernel", entry="read_energy", arguments=["loss"]),
        dict(kind="kernel", entry="backward", arguments=["x", "loss"]),
    ], gradients=dict(x="dx", loss="dloss")))
    module.save(str(directory))
