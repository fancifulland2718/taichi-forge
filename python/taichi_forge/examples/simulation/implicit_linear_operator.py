"""Fixed-topology implicit spring solve with a reusable LinearOperator plan.

This headless example focuses on the execution contract used by implicit
physics workloads: topology is immutable, coefficients change between time
steps, a caller-built block preconditioner is versioned explicitly, and one
SolvePlan is reused on CPU, CUDA, or Vulkan.  The state/RHS boundary uses
compact Vector fields, so qualified backends bind it directly instead of
packing a second solver vector.
"""

import argparse
import math

import numpy as np

import taichi_forge as ti
from taichi_forge.lang import impl


@ti.kernel
def apply_spring_system(
    active_size: ti.i32,
    topology: ti.types.ndarray(dtype=ti.i32, ndim=1),
    numeric: ti.types.ndarray(dtype=ti.f32, ndim=1),
    input: ti.types.ndarray(dtype=ti.f32, ndim=1),
    output: ti.types.ndarray(dtype=ti.f32, ndim=1),
):
    node_count = active_size // 2
    for dof in range(active_size):
        output[dof] = numeric[dof // 2] * input[dof]
    for edge in range(topology[0]):
        first = topology[1 + 2 * edge]
        second = topology[2 + 2 * edge]
        weight = numeric[node_count + edge]
        for axis in ti.static(range(2)):
            difference = input[2 * first + axis] - input[2 * second + axis]
            ti.atomic_add(output[2 * first + axis], weight * difference)
            ti.atomic_add(output[2 * second + axis], -weight * difference)


@ti.data_oriented
class ImplicitSpringChain:
    """A small 2-D linear spring system with fixed connectivity."""

    def __init__(self, node_count=128, dt=1.0 / 120.0):
        if node_count < 3:
            raise ValueError("node_count must be at least 3")
        self.node_count = int(node_count)
        self.dof_count = 2 * self.node_count
        self.edge_count = self.node_count - 1
        self.dt = float(dt)
        # A fixed heterogeneous mass distribution makes the value of the
        # block-diagonal preconditioner visible without changing topology.
        self.mass_values = np.geomspace(
            0.25, 25.0, self.node_count
        ).astype(np.float32)
        self.step_index = 0
        self.operator_numeric_version = 1
        self.preconditioner_numeric_version = 1
        self.last_submission_telemetry = None

        self.displacement = ti.Vector.field(2, ti.f32, shape=self.node_count)
        self.velocity = ti.Vector.field(2, ti.f32, shape=self.node_count)
        self.rhs = ti.Vector.field(2, ti.f32, shape=self.node_count)
        self.next_velocity = ti.Vector.field(2, ti.f32, shape=self.node_count)

        edges = np.column_stack(
            (
                np.arange(self.edge_count, dtype=np.int32),
                np.arange(1, self.node_count, dtype=np.int32),
            )
        )
        self.edge_pairs = ti.ndarray(ti.i32, shape=(self.edge_count, 2))
        self.edge_pairs.from_numpy(edges)
        self.edge_stiffness = ti.ndarray(ti.f32, shape=self.edge_count)
        self.node_mass = ti.ndarray(ti.f32, shape=self.node_count)
        self.node_mass.from_numpy(self.mass_values)

        topology_host = np.empty(1 + 2 * self.edge_count, dtype=np.int32)
        topology_host[0] = self.edge_count
        topology_host[1:] = edges.reshape(-1)
        topology = ti.ndarray(ti.i32, shape=topology_host.size)
        topology.from_numpy(topology_host)
        self.topology = topology

        operator_values, inverse_blocks, edge_values = self._coefficients(0)
        self.operator_numeric = ti.ndarray(ti.f32, shape=operator_values.size)
        self.operator_numeric.from_numpy(operator_values)
        self.inverse_blocks = ti.ndarray(ti.f32, shape=inverse_blocks.size)
        self.inverse_blocks.from_numpy(inverse_blocks)
        self.edge_stiffness.from_numpy(edge_values)

        traits = ti.linalg.OperatorTraits.spd()
        self.operator = ti.linalg.LinearOperator.from_kernel(
            apply_spring_system,
            self.dof_count,
            topology,
            numeric=self.operator_numeric,
            traits=traits,
        )
        self.inverse = ti.linalg.inverse_block_diagonal(
            self.inverse_blocks, block_size=2, assume_spd=True
        )
        self.preconditioner = ti.linalg.experimental.PreconditionerPlan(
            self.operator,
            self.inverse,
            method="implicit_spring_block_diagonal",
        ).setup()
        policy = None
        if impl.current_cfg().arch in (ti.cuda, ti.vulkan):
            policy = "device_convergent"
        self.solve_plan = ti.linalg.experimental.SolvePlan(
            self.operator,
            method="pcg",
            preconditioner=self.preconditioner,
            max_iterations=96,
            atol=1e-7,
            rtol=1e-5,
            execution_policy=policy,
        )
        self.initialize_state()

    def _coefficients(self, step):
        phase = 0.075 * float(step)
        scale = 1.0 + 0.2 * math.sin(phase)
        edge_values = np.asarray(
            [
                scale * (700.0 + 120.0 * ((edge % 5) / 4.0))
                for edge in range(self.edge_count)
            ],
            dtype=np.float32,
        )
        scaled_edges = np.float32(self.dt * self.dt) * edge_values
        operator_values = np.concatenate(
            (
                self.mass_values,
                scaled_edges,
            )
        )
        diagonal = self.mass_values.copy()
        diagonal[:-1] += scaled_edges
        diagonal[1:] += scaled_edges
        inverse_blocks = np.zeros(
            (self.node_count, 2, 2), dtype=np.float32
        )
        inverse_blocks[:, 0, 0] = 1.0 / diagonal
        inverse_blocks[:, 1, 1] = 1.0 / diagonal
        return operator_values, inverse_blocks.reshape(-1), edge_values

    def _publish_coefficients(self, step):
        operator_values, inverse_blocks, edge_values = self._coefficients(step)
        next_operator = ti.ndarray(ti.f32, shape=operator_values.size)
        next_operator.from_numpy(operator_values)
        next_inverse = ti.ndarray(ti.f32, shape=inverse_blocks.size)
        next_inverse.from_numpy(inverse_blocks)
        self.edge_stiffness.from_numpy(edge_values)
        self.operator.update_numeric(
            next_operator,
            expected_topology_version=1,
            expected_numeric_version=self.operator_numeric_version,
        )
        self.inverse.update_numeric(
            next_inverse,
            expected_topology_version=1,
            expected_numeric_version=self.preconditioner_numeric_version,
        )
        self.operator_numeric_version += 1
        self.preconditioner_numeric_version += 1
        self.preconditioner.update()
        self.operator_numeric = next_operator
        self.inverse_blocks = next_inverse

    @ti.kernel
    def initialize_state(self):
        for node in range(self.node_count):
            coordinate = ti.cast(node, ti.f32) / ti.cast(
                self.node_count - 1, ti.f32
            )
            self.displacement[node] = ti.Vector(
                [0.015 * ti.sin(2.0 * math.pi * coordinate), 0.08 * ti.sin(math.pi * coordinate)]
            )
            self.velocity[node] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def assemble_rhs(
        self,
        edge_pairs: ti.types.ndarray(dtype=ti.i32, ndim=2),
        edge_stiffness: ti.types.ndarray(dtype=ti.f32, ndim=1),
        node_mass: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for node in range(self.node_count):
            self.rhs[node] = node_mass[node] * self.velocity[node]
        for edge in range(self.edge_count):
            first = edge_pairs[edge, 0]
            second = edge_pairs[edge, 1]
            difference = self.displacement[first] - self.displacement[second]
            impulse = self.dt * edge_stiffness[edge] * difference
            for axis in ti.static(range(2)):
                ti.atomic_add(self.rhs[first][axis], -impulse[axis])
                ti.atomic_add(self.rhs[second][axis], impulse[axis])

    @ti.kernel
    def advance_state(self):
        for node in range(self.node_count):
            self.velocity[node] = self.next_velocity[node]
            self.displacement[node] += self.dt * self.next_velocity[node]

    def step(self, *, telemetry=False):
        if self.step_index:
            self._publish_coefficients(self.step_index)
        self.assemble_rhs(
            self.edge_pairs, self.edge_stiffness, self.node_mass
        )
        submission = self.solve_plan.submit(
            self.rhs, out=self.next_velocity, telemetry=telemetry
        )
        result = submission.result()
        if telemetry:
            self.last_submission_telemetry = submission.telemetry()
        if not result.converged:
            raise RuntimeError(
                "implicit spring PCG did not converge: "
                f"{result.termination_reason} after {result.iterations} iterations"
            )
        self.advance_state()
        self.step_index += 1
        return result


def _arch_from_name(name):
    return {"cpu": ti.cpu, "cuda": ti.cuda, "vulkan": ti.vulkan}[name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "cuda", "vulkan"), default="cpu")
    parser.add_argument("--nodes", type=int, default=128)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--telemetry", action="store_true")
    args = parser.parse_args()
    ti.init(arch=_arch_from_name(args.arch), offline_cache=False)
    simulation = ImplicitSpringChain(args.nodes)
    iterations = []
    for _ in range(args.steps):
        result = simulation.step(telemetry=args.telemetry)
        iterations.append(result.iterations)
    displacement = simulation.displacement.to_numpy()
    print(
        f"arch={args.arch} steps={args.steps} "
        f"iterations=[{min(iterations)}, {max(iterations)}] "
        f"residual={result.residual_norm:.3e} "
        f"termination={result.termination_reason} "
        f"max_displacement={np.linalg.norm(displacement, axis=1).max():.6f} "
        f"operator_generation={simulation.operator_numeric_version}"
    )
    if args.telemetry:
        telemetry = simulation.last_submission_telemetry
        if telemetry is None:
            print("telemetry=unavailable (CPU submit is synchronously completed)")
        else:
            region = telemetry.regions[0]
            print(
                f"logical={region.logical_iterations} "
                f"encoded={region.encoded_iterations} "
                f"masked={region.masked_iterations} "
                f"backend_graph_launches={telemetry.execution.backend_graph_launches} "
                f"physical_queue_submissions="
                f"{telemetry.execution.physical_queue_submissions}"
            )


if __name__ == "__main__":
    main()
