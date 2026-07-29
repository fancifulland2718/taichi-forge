import numpy as np

import taichi_forge as ti
from tests import test_utils


RUNNING = 0
CONVERGED = 1
BREAKDOWN = 2
USER_STOP = 3


def _dense_operator(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
    size = matrix.shape[0]
    topology = ti.ndarray(ti.i32, shape=size)
    numeric = ti.ndarray(ti.f32, shape=size * size)
    topology.from_numpy(np.arange(size, dtype=np.int32))
    numeric.from_numpy(matrix.reshape(-1))

    @ti.kernel
    def apply_dense(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for row in range(active_size):
            value = 0.0
            for col in range(active_size):
                value += (
                    numeric_data[row * active_size + col]
                    * x[topology_data[col]]
                )
            y[row] = value

    return ti.linalg.LinearOperator.from_kernel(
        apply_dense, size, topology, numeric=numeric
    )


def _diagonal_operator(values):
    values = np.asarray(values, dtype=np.float32)
    size = values.size
    topology = ti.ndarray(ti.i32, shape=size)
    numeric = ti.ndarray(ti.f32, shape=size)
    topology.from_numpy(np.arange(size, dtype=np.int32))
    numeric.from_numpy(values)

    @ti.kernel
    def apply_diagonal(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = numeric_data[index] * x[topology_data[index]]

    return ti.linalg.LinearOperator.from_kernel(
        apply_diagonal, size, topology, numeric=numeric
    )


def _array_arg(name, dtype=ti.f32):
    return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=1)


def _scalar_array_arg(name, dtype):
    return ti.graph.Arg(ti.graph.ArgKind.NDARRAY, name, dtype, ndim=0)


def _scalar_arg(name, dtype):
    return ti.graph.Arg(ti.graph.ArgKind.SCALAR, name, dtype)


def _build_pcg_graph(operator, preconditioner, size, *, max_iterations):
    @ti.kernel
    def initialize(
        b: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        r: ti.types.ndarray(dtype=ti.f32, ndim=1),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        status: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        predicate[None] = 0
        status[None] = RUNNING
        counter[None] = 0
        for index in range(size):
            x[index] = 0.0
            r[index] = b[index]

    @ti.kernel
    def seed_direction(
        b: ti.types.ndarray(dtype=ti.f32, ndim=1),
        r: ti.types.ndarray(dtype=ti.f32, ndim=1),
        z: ti.types.ndarray(dtype=ti.f32, ndim=1),
        p: ti.types.ndarray(dtype=ti.f32, ndim=1),
        residual_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
        norm_b_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
        rz_old: ti.types.ndarray(dtype=ti.f32, ndim=0),
    ):
        residual_sq[None] = 0.0
        norm_b_sq[None] = 0.0
        rz_old[None] = 0.0
        for index in range(size):
            p[index] = z[index]
            ti.atomic_add(residual_sq[None], r[index] * r[index])
            ti.atomic_add(norm_b_sq[None], b[index] * b[index])
            ti.atomic_add(rz_old[None], r[index] * z[index])

    @ti.kernel
    def evaluate_condition(
        residual_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
        norm_b_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        status: ti.types.ndarray(dtype=ti.i32, ndim=0),
        user_stop: ti.types.ndarray(dtype=ti.i32, ndim=0),
        atol: ti.f32,
        rtol: ti.f32,
    ):
        if status[None] == RUNNING:
            threshold = ti.max(
                atol, rtol * ti.sqrt(ti.max(norm_b_sq[None], 0.0))
            )
            if user_stop[None] != 0:
                status[None] = USER_STOP
            elif residual_sq[None] <= threshold * threshold:
                status[None] = CONVERGED
        predicate[None] = int(status[None] == RUNNING)

    @ti.kernel
    def reduce_pap(
        p: ti.types.ndarray(dtype=ti.f32, ndim=1),
        ap: ti.types.ndarray(dtype=ti.f32, ndim=1),
        pap: ti.types.ndarray(dtype=ti.f32, ndim=0),
    ):
        pap[None] = 0.0
        for index in range(size):
            ti.atomic_add(pap[None], p[index] * ap[index])

    @ti.kernel
    def update_solution_residual(
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        r: ti.types.ndarray(dtype=ti.f32, ndim=1),
        p: ti.types.ndarray(dtype=ti.f32, ndim=1),
        ap: ti.types.ndarray(dtype=ti.f32, ndim=1),
        rz_old: ti.types.ndarray(dtype=ti.f32, ndim=0),
        pap: ti.types.ndarray(dtype=ti.f32, ndim=0),
        alpha: ti.types.ndarray(dtype=ti.f32, ndim=0),
        status: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        if status[None] == RUNNING:
            if ti.abs(pap[None]) <= 1.0e-20:
                status[None] = BREAKDOWN
            else:
                alpha[None] = rz_old[None] / pap[None]
        if status[None] == RUNNING:
            for index in range(size):
                x[index] += alpha[None] * p[index]
                r[index] -= alpha[None] * ap[index]

    @ti.kernel
    def reduce_next(
        r: ti.types.ndarray(dtype=ti.f32, ndim=1),
        z: ti.types.ndarray(dtype=ti.f32, ndim=1),
        residual_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
        rz_new: ti.types.ndarray(dtype=ti.f32, ndim=0),
    ):
        residual_sq[None] = 0.0
        rz_new[None] = 0.0
        for index in range(size):
            ti.atomic_add(residual_sq[None], r[index] * r[index])
            ti.atomic_add(rz_new[None], r[index] * z[index])

    @ti.kernel
    def finish_iteration(
        z: ti.types.ndarray(dtype=ti.f32, ndim=1),
        p: ti.types.ndarray(dtype=ti.f32, ndim=1),
        rz_old: ti.types.ndarray(dtype=ti.f32, ndim=0),
        rz_new: ti.types.ndarray(dtype=ti.f32, ndim=0),
        beta: ti.types.ndarray(dtype=ti.f32, ndim=0),
        status: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        if status[None] == RUNNING:
            if ti.abs(rz_old[None]) <= 1.0e-20:
                status[None] = BREAKDOWN
            else:
                beta[None] = rz_new[None] / rz_old[None]
                rz_old[None] = rz_new[None]
                counter[None] += 1
        if status[None] == RUNNING:
            for index in range(size):
                p[index] = z[index] + beta[None] * p[index]

    vectors = {
        name: _array_arg(name)
        for name in ("b", "x", "r", "z", "p", "ap")
    }
    scalars = {
        name: _scalar_array_arg(name, ti.f32)
        for name in (
            "residual_sq",
            "norm_b_sq",
            "rz_old",
            "rz_new",
            "pap",
            "alpha",
            "beta",
        )
    }
    predicate = _scalar_array_arg("predicate", ti.i32)
    status = _scalar_array_arg("status", ti.i32)
    counter = _scalar_array_arg("counter", ti.i32)
    user_stop = _scalar_array_arg("user_stop", ti.i32)
    atol = _scalar_arg("atol", ti.f32)
    rtol = _scalar_arg("rtol", ti.f32)

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        initialize,
        vectors["b"],
        vectors["x"],
        vectors["r"],
        predicate,
        status,
        counter,
    )
    builder.append_native(
        preconditioner.graph_action(vectors["r"], vectors["z"])
    )
    builder.dispatch(
        seed_direction,
        vectors["b"],
        vectors["r"],
        vectors["z"],
        vectors["p"],
        scalars["residual_sq"],
        scalars["norm_b_sq"],
        scalars["rz_old"],
    )

    condition = builder.create_sequential()
    condition.dispatch(
        evaluate_condition,
        scalars["residual_sq"],
        scalars["norm_b_sq"],
        predicate,
        status,
        user_stop,
        atol,
        rtol,
    )
    body = builder.create_sequential()
    body.append_native(operator.graph_action(vectors["p"], vectors["ap"]))
    body.dispatch(reduce_pap, vectors["p"], vectors["ap"], scalars["pap"])
    body.dispatch(
        update_solution_residual,
        vectors["x"],
        vectors["r"],
        vectors["p"],
        vectors["ap"],
        scalars["rz_old"],
        scalars["pap"],
        scalars["alpha"],
        status,
    )
    body.append_native(
        preconditioner.graph_action(vectors["r"], vectors["z"])
    )
    body.dispatch(
        reduce_next,
        vectors["r"],
        vectors["z"],
        scalars["residual_sq"],
        scalars["rz_new"],
    )
    body.dispatch(
        finish_iteration,
        vectors["z"],
        vectors["p"],
        scalars["rz_old"],
        scalars["rz_new"],
        scalars["beta"],
        status,
        counter,
    )
    builder.while_loop(
        condition,
        body,
        predicate=predicate,
        status=status,
        control_inputs=(
            scalars["residual_sq"],
            scalars["norm_b_sq"],
            user_stop,
            atol,
            rtol,
        ),
        carried_state=(
            vectors["x"],
            vectors["r"],
            vectors["z"],
            vectors["p"],
            vectors["ap"],
            scalars["residual_sq"],
            scalars["rz_old"],
            scalars["rz_new"],
        ),
        counter=counter,
        max_iterations=max_iterations,
        name="qualified_pcg",
    )
    return builder.compile()


def _build_bicgstab_graph(operator, size, *, max_iterations):
    @ti.kernel
    def initialize(
        b: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        r: ti.types.ndarray(dtype=ti.f32, ndim=1),
        r_hat: ti.types.ndarray(dtype=ti.f32, ndim=1),
        p: ti.types.ndarray(dtype=ti.f32, ndim=1),
        v: ti.types.ndarray(dtype=ti.f32, ndim=1),
        residual_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
        norm_b_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
        rho_old: ti.types.ndarray(dtype=ti.f32, ndim=0),
        alpha: ti.types.ndarray(dtype=ti.f32, ndim=0),
        omega: ti.types.ndarray(dtype=ti.f32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        status: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        residual_sq[None] = 0.0
        norm_b_sq[None] = 0.0
        rho_old[None] = 1.0
        alpha[None] = 1.0
        omega[None] = 1.0
        predicate[None] = 0
        status[None] = RUNNING
        counter[None] = 0
        for index in range(size):
            x[index] = 0.0
            r[index] = b[index]
            r_hat[index] = b[index]
            p[index] = 0.0
            v[index] = 0.0
            ti.atomic_add(residual_sq[None], b[index] * b[index])
            ti.atomic_add(norm_b_sq[None], b[index] * b[index])

    @ti.kernel
    def evaluate_condition(
        residual_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
        norm_b_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
        predicate: ti.types.ndarray(dtype=ti.i32, ndim=0),
        status: ti.types.ndarray(dtype=ti.i32, ndim=0),
        user_stop: ti.types.ndarray(dtype=ti.i32, ndim=0),
        atol: ti.f32,
        rtol: ti.f32,
    ):
        if status[None] == RUNNING:
            threshold = ti.max(
                atol, rtol * ti.sqrt(ti.max(norm_b_sq[None], 0.0))
            )
            if user_stop[None] != 0:
                status[None] = USER_STOP
            elif residual_sq[None] <= threshold * threshold:
                status[None] = CONVERGED
        predicate[None] = int(status[None] == RUNNING)

    @ti.kernel
    def update_search_direction(
        r: ti.types.ndarray(dtype=ti.f32, ndim=1),
        r_hat: ti.types.ndarray(dtype=ti.f32, ndim=1),
        p: ti.types.ndarray(dtype=ti.f32, ndim=1),
        v: ti.types.ndarray(dtype=ti.f32, ndim=1),
        rho_old: ti.types.ndarray(dtype=ti.f32, ndim=0),
        rho_new: ti.types.ndarray(dtype=ti.f32, ndim=0),
        alpha: ti.types.ndarray(dtype=ti.f32, ndim=0),
        omega: ti.types.ndarray(dtype=ti.f32, ndim=0),
        beta: ti.types.ndarray(dtype=ti.f32, ndim=0),
        status: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        rho_new[None] = 0.0
        for index in range(size):
            ti.atomic_add(rho_new[None], r_hat[index] * r[index])
        if status[None] == RUNNING:
            if (
                ti.abs(rho_old[None]) <= 1.0e-20
                or ti.abs(omega[None]) <= 1.0e-20
                or ti.abs(rho_new[None]) <= 1.0e-20
            ):
                status[None] = BREAKDOWN
            else:
                beta[None] = (
                    rho_new[None] / rho_old[None]
                ) * (alpha[None] / omega[None])
        if status[None] == RUNNING:
            for index in range(size):
                p[index] = r[index] + beta[None] * (
                    p[index] - omega[None] * v[index]
                )

    @ti.kernel
    def form_intermediate_residual(
        r: ti.types.ndarray(dtype=ti.f32, ndim=1),
        r_hat: ti.types.ndarray(dtype=ti.f32, ndim=1),
        p: ti.types.ndarray(dtype=ti.f32, ndim=1),
        v: ti.types.ndarray(dtype=ti.f32, ndim=1),
        s: ti.types.ndarray(dtype=ti.f32, ndim=1),
        rho_new: ti.types.ndarray(dtype=ti.f32, ndim=0),
        denominator: ti.types.ndarray(dtype=ti.f32, ndim=0),
        alpha: ti.types.ndarray(dtype=ti.f32, ndim=0),
        s_norm_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
        status: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        denominator[None] = 0.0
        s_norm_sq[None] = 0.0
        for index in range(size):
            ti.atomic_add(denominator[None], r_hat[index] * v[index])
        if status[None] == RUNNING:
            if ti.abs(denominator[None]) <= 1.0e-20:
                status[None] = BREAKDOWN
            else:
                alpha[None] = rho_new[None] / denominator[None]
        if status[None] == RUNNING:
            for index in range(size):
                s[index] = r[index] - alpha[None] * v[index]
                ti.atomic_add(s_norm_sq[None], s[index] * s[index])

    @ti.kernel
    def accept_intermediate_convergence(
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        r: ti.types.ndarray(dtype=ti.f32, ndim=1),
        p: ti.types.ndarray(dtype=ti.f32, ndim=1),
        s: ti.types.ndarray(dtype=ti.f32, ndim=1),
        alpha: ti.types.ndarray(dtype=ti.f32, ndim=0),
        s_norm_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
        norm_b_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
        residual_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
        status: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
        atol: ti.f32,
        rtol: ti.f32,
    ):
        if status[None] == RUNNING:
            threshold = ti.max(
                atol, rtol * ti.sqrt(ti.max(norm_b_sq[None], 0.0))
            )
            if s_norm_sq[None] <= threshold * threshold:
                status[None] = CONVERGED
                residual_sq[None] = s_norm_sq[None]
                counter[None] += 1
        if status[None] == CONVERGED:
            for index in range(size):
                x[index] += alpha[None] * p[index]
                r[index] = s[index]

    @ti.kernel
    def finish_iteration(
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        r: ti.types.ndarray(dtype=ti.f32, ndim=1),
        p: ti.types.ndarray(dtype=ti.f32, ndim=1),
        s: ti.types.ndarray(dtype=ti.f32, ndim=1),
        t: ti.types.ndarray(dtype=ti.f32, ndim=1),
        rho_old: ti.types.ndarray(dtype=ti.f32, ndim=0),
        rho_new: ti.types.ndarray(dtype=ti.f32, ndim=0),
        alpha: ti.types.ndarray(dtype=ti.f32, ndim=0),
        omega: ti.types.ndarray(dtype=ti.f32, ndim=0),
        residual_sq: ti.types.ndarray(dtype=ti.f32, ndim=0),
        status: ti.types.ndarray(dtype=ti.i32, ndim=0),
        counter: ti.types.ndarray(dtype=ti.i32, ndim=0),
    ):
        ts = 0.0
        tt = 0.0
        for index in range(size):
            ts += t[index] * s[index]
            tt += t[index] * t[index]
        if status[None] == RUNNING:
            if ti.abs(tt) <= 1.0e-20:
                status[None] = BREAKDOWN
            else:
                omega[None] = ts / tt
                if ti.abs(omega[None]) <= 1.0e-20:
                    status[None] = BREAKDOWN
        if status[None] == RUNNING:
            residual_sq[None] = 0.0
            for index in range(size):
                x[index] += alpha[None] * p[index] + omega[None] * s[index]
                r[index] = s[index] - omega[None] * t[index]
                ti.atomic_add(residual_sq[None], r[index] * r[index])
            rho_old[None] = rho_new[None]
            counter[None] += 1

    vectors = {
        name: _array_arg(name)
        for name in ("b", "x", "r", "r_hat", "p", "v", "s", "t")
    }
    scalars = {
        name: _scalar_array_arg(name, ti.f32)
        for name in (
            "residual_sq",
            "norm_b_sq",
            "rho_old",
            "rho_new",
            "alpha",
            "omega",
            "beta",
            "denominator",
            "s_norm_sq",
        )
    }
    predicate = _scalar_array_arg("predicate", ti.i32)
    status = _scalar_array_arg("status", ti.i32)
    counter = _scalar_array_arg("counter", ti.i32)
    user_stop = _scalar_array_arg("user_stop", ti.i32)
    atol = _scalar_arg("atol", ti.f32)
    rtol = _scalar_arg("rtol", ti.f32)

    builder = ti.graph.GraphBuilder()
    builder.dispatch(
        initialize,
        vectors["b"],
        vectors["x"],
        vectors["r"],
        vectors["r_hat"],
        vectors["p"],
        vectors["v"],
        scalars["residual_sq"],
        scalars["norm_b_sq"],
        scalars["rho_old"],
        scalars["alpha"],
        scalars["omega"],
        predicate,
        status,
        counter,
    )
    condition = builder.create_sequential()
    condition.dispatch(
        evaluate_condition,
        scalars["residual_sq"],
        scalars["norm_b_sq"],
        predicate,
        status,
        user_stop,
        atol,
        rtol,
    )
    body = builder.create_sequential()
    body.dispatch(
        update_search_direction,
        vectors["r"],
        vectors["r_hat"],
        vectors["p"],
        vectors["v"],
        scalars["rho_old"],
        scalars["rho_new"],
        scalars["alpha"],
        scalars["omega"],
        scalars["beta"],
        status,
    )
    body.append_native(operator.graph_action(vectors["p"], vectors["v"]))
    body.dispatch(
        form_intermediate_residual,
        vectors["r"],
        vectors["r_hat"],
        vectors["p"],
        vectors["v"],
        vectors["s"],
        scalars["rho_new"],
        scalars["denominator"],
        scalars["alpha"],
        scalars["s_norm_sq"],
        status,
    )
    body.dispatch(
        accept_intermediate_convergence,
        vectors["x"],
        vectors["r"],
        vectors["p"],
        vectors["s"],
        scalars["alpha"],
        scalars["s_norm_sq"],
        scalars["norm_b_sq"],
        scalars["residual_sq"],
        status,
        counter,
        atol,
        rtol,
    )
    body.append_native(operator.graph_action(vectors["s"], vectors["t"]))
    body.dispatch(
        finish_iteration,
        vectors["x"],
        vectors["r"],
        vectors["p"],
        vectors["s"],
        vectors["t"],
        scalars["rho_old"],
        scalars["rho_new"],
        scalars["alpha"],
        scalars["omega"],
        scalars["residual_sq"],
        status,
        counter,
    )
    builder.while_loop(
        condition,
        body,
        predicate=predicate,
        status=status,
        control_inputs=(
            scalars["residual_sq"],
            scalars["norm_b_sq"],
            user_stop,
            atol,
            rtol,
        ),
        carried_state=(
            vectors["x"],
            vectors["r"],
            vectors["p"],
            vectors["v"],
            vectors["s"],
            vectors["t"],
            scalars["residual_sq"],
            scalars["rho_old"],
            scalars["rho_new"],
            scalars["alpha"],
            scalars["omega"],
        ),
        counter=counter,
        max_iterations=max_iterations,
        name="qualified_bicgstab",
    )
    return builder.compile()


def _runtime_arguments(vector_names, scalar_names, rhs):
    arguments = {
        name: ti.ndarray(ti.f32, shape=rhs.size) for name in vector_names
    }
    arguments.update(
        {name: ti.ndarray(ti.f32, shape=()) for name in scalar_names}
    )
    arguments.update(
        {
            "predicate": ti.ndarray(ti.i32, shape=()),
            "status": ti.ndarray(ti.i32, shape=()),
            "counter": ti.ndarray(ti.i32, shape=()),
            "user_stop": ti.ndarray(ti.i32, shape=()),
        }
    )
    arguments["b"].from_numpy(np.asarray(rhs, dtype=np.float32))
    arguments["user_stop"].fill(0)
    arguments["atol"] = 1.0e-5
    arguments["rtol"] = 1.0e-5
    return arguments


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_generic_structured_graph_qualifies_preconditioned_cg():
    matrix = np.asarray(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    rhs = np.asarray([1.0, 2.0, -3.0, 4.0], dtype=np.float32)
    operator = _dense_operator(matrix)
    preconditioner = _diagonal_operator(1.0 / np.diag(matrix))
    graph = _build_pcg_graph(
        operator, preconditioner, rhs.size, max_iterations=16
    )
    arguments = _runtime_arguments(
        ("b", "x", "r", "z", "p", "ap"),
        (
            "residual_sq",
            "norm_b_sq",
            "rz_old",
            "rz_new",
            "pap",
            "alpha",
            "beta",
        ),
        rhs,
    )

    graph.run(arguments)
    np.testing.assert_allclose(
        arguments["x"].to_numpy(),
        np.linalg.solve(matrix, rhs),
        rtol=3.0e-4,
        atol=3.0e-5,
    )
    report = graph.control_flow_stats()[0]
    assert report.name == "qualified_pcg"
    assert report.initial_status == RUNNING
    assert report.final_status == CONVERGED
    assert 1 <= report.logical_iterations <= rhs.size
    assert graph._debug_info["native_count"] == 3

    arguments["user_stop"].fill(1)
    graph.run(arguments)
    report = graph.control_flow_stats()[0]
    assert report.logical_iterations == 0
    assert report.final_status == USER_STOP
    np.testing.assert_array_equal(
        arguments["x"].to_numpy(), np.zeros(rhs.size, dtype=np.float32)
    )


@test_utils.test(arch=[ti.cpu, ti.cuda, ti.vulkan], offline_cache=False)
def test_generic_structured_graph_qualifies_nonsymmetric_bicgstab():
    matrix = np.asarray(
        [
            [4.0, 1.0, 0.0, 0.0],
            [-2.0, 3.0, 1.0, 0.0],
            [0.0, -1.0, 2.0, 1.0],
            [1.0, 0.0, -1.0, 2.0],
        ],
        dtype=np.float32,
    )
    assert not np.allclose(matrix, matrix.T)
    rhs = np.asarray([1.0, -2.0, 3.0, 0.5], dtype=np.float32)
    operator = _dense_operator(matrix)
    graph = _build_bicgstab_graph(operator, rhs.size, max_iterations=16)
    arguments = _runtime_arguments(
        ("b", "x", "r", "r_hat", "p", "v", "s", "t"),
        (
            "residual_sq",
            "norm_b_sq",
            "rho_old",
            "rho_new",
            "alpha",
            "omega",
            "beta",
            "denominator",
            "s_norm_sq",
        ),
        rhs,
    )

    graph.run(arguments)
    np.testing.assert_allclose(
        arguments["x"].to_numpy(),
        np.linalg.solve(matrix, rhs),
        rtol=5.0e-4,
        atol=5.0e-5,
    )
    report = graph.control_flow_stats()[0]
    assert report.name == "qualified_bicgstab"
    assert report.initial_status == RUNNING
    assert report.final_status == CONVERGED
    assert 1 <= report.logical_iterations <= rhs.size
    assert graph._debug_info["native_count"] == 2
