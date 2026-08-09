"""Backend-portable recurrence kernels for independent batched CG/PCG."""

from taichi_forge.lang import ops
from taichi_forge.lang.device_extent import device_extent_count
from taichi_forge.lang.kernel_impl import kernel
from taichi_forge.types import ndarray_type
from taichi_forge.types.annotations import template
from taichi_forge.types.primitive_types import f32, i32

RR_CURRENT = 0
RR_NEXT = 1
RHO_CURRENT = 2
RHO_NEXT = 3
P_AP = 4
ALPHA = 5
BETA = 6
TOLERANCE_SQUARED = 7
RHS_SQUARED = 8
INITIAL_RR = 9
REFERENCE_NORM = 10
EFFECTIVE_TOLERANCE = 11
FLOAT_STATE_SLOTS = 12

STATUS = 0
ITERATIONS = 1
ACTIVE = 2
INT_STATE_SLOTS = 3

ACTIVE_COUNT = 0
EXECUTED_SYSTEM_ITERATIONS = 1
COUNTER_SLOTS = 2

TERMINAL_SCHEMA_VERSION = 1
TERMINAL_HEADER_SLOTS = 4
TERMINAL_STATUS = 0
TERMINAL_ITERATIONS = 1
TERMINAL_INITIAL_RR = 2
TERMINAL_FINAL_RR = 3
TERMINAL_REFERENCE_NORM = 4
TERMINAL_EFFECTIVE_TOLERANCE = 5
TERMINAL_SYSTEM_SLOTS = 6


@kernel
def initialize_output(
    initial: ndarray_type.ndarray(dtype=f32, ndim=1),
    output: ndarray_type.ndarray(dtype=f32, ndim=1),
    use_initial_guess: i32,
    total_size: i32,
):
    for index in range(total_size):
        output[index] = initial[index] if use_initial_guess != 0 else 0.0


@kernel
def initialize_loop_control(
    predicate: ndarray_type.ndarray(dtype=i32, ndim=0),
    status: ndarray_type.ndarray(dtype=i32, ndim=0),
    counter: ndarray_type.ndarray(dtype=i32, ndim=0),
):
    predicate[None] = 0
    status[None] = 0
    counter[None] = 0


@kernel
def evaluate_active_systems(
    counters: ndarray_type.ndarray(dtype=i32, ndim=1),
    predicate: ndarray_type.ndarray(dtype=i32, ndim=0),
    status: ndarray_type.ndarray(dtype=i32, ndim=0),
):
    active = counters[ACTIVE_COUNT]
    predicate[None] = int(active > 0)
    status[None] = int(active <= 0)


@kernel
def publish_active_system_extent(
    int_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    active_systems: ndarray_type.ndarray(dtype=i32, ndim=1),
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    float_state: ndarray_type.ndarray(dtype=f32, ndim=1),
    system_size: i32,
    batch_size: i32,
):
    for _ in range(1):
        active_count = 0
        for env in range(batch_size):
            float_state[P_AP * batch_size + env] = 0.0
            if int_state[ACTIVE * batch_size + env] != 0:
                active_systems[active_count] = env
                active_count += 1
        extent_state[0] = active_count * system_size
        extent_state[1] = 0


@kernel
def advance_loop_counter(
    counter: ndarray_type.ndarray(dtype=i32, ndim=0),
):
    counter[None] += 1


@kernel
def publish_terminal_packet_device(
    float_state: ndarray_type.ndarray(dtype=f32, ndim=1),
    int_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    counters: ndarray_type.ndarray(dtype=i32, ndim=1),
    logical_counter: ndarray_type.ndarray(dtype=i32, ndim=0),
    packet: ndarray_type.ndarray(dtype=i32, ndim=1),
    batch_size: i32,
):
    for _ in range(1):
        packet[0] = TERMINAL_SCHEMA_VERSION
        packet[1] = logical_counter[None]
        packet[2] = counters[EXECUTED_SYSTEM_ITERATIONS]
        packet[3] = counters[ACTIVE_COUNT]
    for env in range(batch_size):
        base = TERMINAL_HEADER_SLOTS + env * TERMINAL_SYSTEM_SLOTS
        packet[base + TERMINAL_STATUS] = int_state[STATUS * batch_size + env]
        packet[base + TERMINAL_ITERATIONS] = int_state[
            ITERATIONS * batch_size + env
        ]
        packet[base + TERMINAL_INITIAL_RR] = ops.bit_cast(
            float_state[INITIAL_RR * batch_size + env], i32
        )
        packet[base + TERMINAL_FINAL_RR] = ops.bit_cast(
            float_state[RR_CURRENT * batch_size + env], i32
        )
        packet[base + TERMINAL_REFERENCE_NORM] = ops.bit_cast(
            float_state[REFERENCE_NORM * batch_size + env], i32
        )
        packet[base + TERMINAL_EFFECTIVE_TOLERANCE] = ops.bit_cast(
            float_state[EFFECTIVE_TOLERANCE * batch_size + env], i32
        )


@kernel
def publish_terminal_packet_host_count(
    float_state: ndarray_type.ndarray(dtype=f32, ndim=1),
    int_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    counters: ndarray_type.ndarray(dtype=i32, ndim=1),
    logical_iterations: i32,
    packet: ndarray_type.ndarray(dtype=i32, ndim=1),
    batch_size: i32,
):
    for _ in range(1):
        packet[0] = TERMINAL_SCHEMA_VERSION
        packet[1] = logical_iterations
        packet[2] = counters[EXECUTED_SYSTEM_ITERATIONS]
        packet[3] = counters[ACTIVE_COUNT]
    for env in range(batch_size):
        base = TERMINAL_HEADER_SLOTS + env * TERMINAL_SYSTEM_SLOTS
        packet[base + TERMINAL_STATUS] = int_state[STATUS * batch_size + env]
        packet[base + TERMINAL_ITERATIONS] = int_state[
            ITERATIONS * batch_size + env
        ]
        packet[base + TERMINAL_INITIAL_RR] = ops.bit_cast(
            float_state[INITIAL_RR * batch_size + env], i32
        )
        packet[base + TERMINAL_FINAL_RR] = ops.bit_cast(
            float_state[RR_CURRENT * batch_size + env], i32
        )
        packet[base + TERMINAL_REFERENCE_NORM] = ops.bit_cast(
            float_state[REFERENCE_NORM * batch_size + env], i32
        )
        packet[base + TERMINAL_EFFECTIVE_TOLERANCE] = ops.bit_cast(
            float_state[EFFECTIVE_TOLERANCE * batch_size + env], i32
        )


@kernel
def initialize_residual(
    rhs: ndarray_type.ndarray(dtype=f32, ndim=1),
    applied: ndarray_type.ndarray(dtype=f32, ndim=1),
    residual: ndarray_type.ndarray(dtype=f32, ndim=1),
    float_state: ndarray_type.ndarray(dtype=f32, ndim=1),
    int_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    absolute_tolerance: ndarray_type.ndarray(dtype=f32, ndim=1),
    relative_tolerance: ndarray_type.ndarray(dtype=f32, ndim=1),
    counters: ndarray_type.ndarray(dtype=i32, ndim=1),
    total_size: i32,
    system_size: i32,
    batch_size: i32,
):
    for env in range(batch_size):
        for slot in range(FLOAT_STATE_SLOTS):
            float_state[slot * batch_size + env] = 0.0
        int_state[STATUS * batch_size + env] = 0
        int_state[ITERATIONS * batch_size + env] = 0
        int_state[ACTIVE * batch_size + env] = 0
    for slot in range(COUNTER_SLOTS):
        counters[slot] = 0
    for index in range(total_size):
        env = index // system_size
        value = rhs[index] - applied[index]
        residual[index] = value
        ops.atomic_add(float_state[RR_CURRENT * batch_size + env],
                       value * value)
        ops.atomic_add(
            float_state[RHS_SQUARED * batch_size + env],
            rhs[index] * rhs[index],
        )
    for env in range(batch_size):
        rr = float_state[RR_CURRENT * batch_size + env]
        rhs_squared = float_state[RHS_SQUARED * batch_size + env]
        reference = 0.0
        if relative_tolerance[env] > 0.0:
            reference = ops.sqrt(rhs_squared)
        effective = ops.max(absolute_tolerance[env],
                            relative_tolerance[env] * reference)
        tolerance_squared = effective * effective
        float_state[INITIAL_RR * batch_size + env] = rr
        float_state[REFERENCE_NORM * batch_size + env] = reference
        float_state[EFFECTIVE_TOLERANCE * batch_size + env] = effective
        float_state[TOLERANCE_SQUARED * batch_size + env] = tolerance_squared
        invalid = (rr != rr or rr < 0.0 or rr > 3.402823466e38
                   or rhs_squared != rhs_squared or rhs_squared < 0.0
                   or rhs_squared > 3.402823466e38 or effective != effective
                   or effective > 3.402823466e38
                   or tolerance_squared != tolerance_squared
                   or tolerance_squared > 3.402823466e38)
        if invalid:
            int_state[STATUS * batch_size + env] = 1
        elif rr <= tolerance_squared:
            int_state[STATUS * batch_size + env] = 2
        else:
            int_state[ACTIVE * batch_size + env] = 1
            ops.atomic_add(counters[ACTIVE_COUNT], 1)


@kernel
def reduce_dot(
        left: ndarray_type.ndarray(dtype=f32, ndim=1),
        right: ndarray_type.ndarray(dtype=f32, ndim=1),
        float_state: ndarray_type.ndarray(dtype=f32, ndim=1),
        int_state: ndarray_type.ndarray(dtype=i32, ndim=1),
        total_size: i32,
        system_size: i32,
        batch_size: i32,
        state_slot: template(),
):
    for env in range(batch_size):
        float_state[state_slot * batch_size + env] = 0.0
    for index in range(total_size):
        env = index // system_size
        if int_state[ACTIVE * batch_size + env] != 0:
            ops.atomic_add(
                float_state[state_slot * batch_size + env],
                left[index] * right[index],
            )


@kernel
def reduce_dot_compact(
        left: ndarray_type.ndarray(dtype=f32, ndim=1),
        right: ndarray_type.ndarray(dtype=f32, ndim=1),
        float_state: ndarray_type.ndarray(dtype=f32, ndim=1),
        int_state: ndarray_type.ndarray(dtype=i32, ndim=1),
        active_systems: ndarray_type.ndarray(dtype=i32, ndim=1),
        extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
        total_size: template(),
        system_size: i32,
        batch_size: i32,
        state_slot: template(),
):
    for compact_index in range(total_size):
        if compact_index < device_extent_count(extent_state):
            active_slot = compact_index // system_size
            local_index = compact_index - active_slot * system_size
            env = active_systems[active_slot]
            if int_state[ACTIVE * batch_size + env] != 0:
                index = env * system_size + local_index
                ops.atomic_add(
                    float_state[state_slot * batch_size + env],
                    left[index] * right[index],
                )


@kernel
def validate_initial_rho(
        float_state: ndarray_type.ndarray(dtype=f32, ndim=1),
        int_state: ndarray_type.ndarray(dtype=i32, ndim=1),
        counters: ndarray_type.ndarray(dtype=i32, ndim=1),
        batch_size: i32,
):
    for env in range(batch_size):
        if int_state[ACTIVE * batch_size + env] != 0:
            rho = float_state[RHO_CURRENT * batch_size + env]
            if rho != rho or rho <= 0.0 or rho > 3.402823466e38:
                int_state[STATUS * batch_size + env] = 1
                int_state[ACTIVE * batch_size + env] = 0
                ops.atomic_sub(counters[ACTIVE_COUNT], 1)


@kernel
def initialize_direction(
    source: ndarray_type.ndarray(dtype=f32, ndim=1),
    direction: ndarray_type.ndarray(dtype=f32, ndim=1),
    int_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    total_size: i32,
    system_size: i32,
    batch_size: i32,
):
    for index in range(total_size):
        env = index // system_size
        if int_state[ACTIVE * batch_size + env] != 0:
            direction[index] = source[index]
        else:
            direction[index] = 0.0


@kernel
def prepare_alpha(
        float_state: ndarray_type.ndarray(dtype=f32, ndim=1),
        int_state: ndarray_type.ndarray(dtype=i32, ndim=1),
        counters: ndarray_type.ndarray(dtype=i32, ndim=1),
        batch_size: i32,
        preconditioned: template(),
):
    for env in range(batch_size):
        float_state[RR_NEXT * batch_size + env] = 0.0
        if int_state[ACTIVE * batch_size + env] != 0:
            denominator = float_state[P_AP * batch_size + env]
            numerator = float_state[
                (RHO_CURRENT if preconditioned else RR_CURRENT) * batch_size +
                env]
            alpha = numerator / denominator
            if (denominator != denominator or denominator <= 0.0
                    or denominator > 3.402823466e38 or alpha != alpha
                    or alpha > 3.402823466e38 or alpha < -3.402823466e38):
                int_state[STATUS * batch_size + env] = 1
                int_state[ACTIVE * batch_size + env] = 0
                ops.atomic_sub(counters[ACTIVE_COUNT], 1)
                alpha = 0.0
            float_state[ALPHA * batch_size + env] = alpha
        else:
            float_state[ALPHA * batch_size + env] = 0.0


@kernel
def update_solution_residual(
    direction: ndarray_type.ndarray(dtype=f32, ndim=1),
    applied: ndarray_type.ndarray(dtype=f32, ndim=1),
    solution: ndarray_type.ndarray(dtype=f32, ndim=1),
    residual: ndarray_type.ndarray(dtype=f32, ndim=1),
    float_state: ndarray_type.ndarray(dtype=f32, ndim=1),
    int_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    counters: ndarray_type.ndarray(dtype=i32, ndim=1),
    total_size: i32,
    system_size: i32,
    batch_size: i32,
):
    for env in range(batch_size):
        float_state[RR_NEXT * batch_size + env] = 0.0
    for index in range(total_size):
        env = index // system_size
        if int_state[ACTIVE * batch_size + env] != 0:
            alpha = float_state[ALPHA * batch_size + env]
            solution[index] += alpha * direction[index]
            value = residual[index] - alpha * applied[index]
            residual[index] = value
            ops.atomic_add(float_state[RR_NEXT * batch_size + env],
                           value * value)
    for env in range(batch_size):
        if int_state[ACTIVE * batch_size + env] != 0:
            rr_next = float_state[RR_NEXT * batch_size + env]
            int_state[ITERATIONS * batch_size + env] += 1
            ops.atomic_add(counters[EXECUTED_SYSTEM_ITERATIONS], 1)
            invalid = (rr_next != rr_next or rr_next < 0.0
                       or rr_next > 3.402823466e38)
            if invalid:
                float_state[RR_CURRENT * batch_size + env] = rr_next
                int_state[STATUS * batch_size + env] = 1
                int_state[ACTIVE * batch_size + env] = 0
                ops.atomic_sub(counters[ACTIVE_COUNT], 1)
            elif rr_next <= float_state[TOLERANCE_SQUARED * batch_size + env]:
                float_state[RR_CURRENT * batch_size + env] = rr_next
                int_state[STATUS * batch_size + env] = 2
                int_state[ACTIVE * batch_size + env] = 0
                ops.atomic_sub(counters[ACTIVE_COUNT], 1)


@kernel
def update_solution_residual_compact_values(
    direction: ndarray_type.ndarray(dtype=f32, ndim=1),
    applied: ndarray_type.ndarray(dtype=f32, ndim=1),
    solution: ndarray_type.ndarray(dtype=f32, ndim=1),
    residual: ndarray_type.ndarray(dtype=f32, ndim=1),
    float_state: ndarray_type.ndarray(dtype=f32, ndim=1),
    int_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    active_systems: ndarray_type.ndarray(dtype=i32, ndim=1),
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    total_size: template(),
    system_size: i32,
    batch_size: i32,
):
    for compact_index in range(total_size):
        if compact_index < device_extent_count(extent_state):
            active_slot = compact_index // system_size
            local_index = compact_index - active_slot * system_size
            env = active_systems[active_slot]
            if int_state[ACTIVE * batch_size + env] != 0:
                index = env * system_size + local_index
                alpha = float_state[ALPHA * batch_size + env]
                solution[index] += alpha * direction[index]
                value = residual[index] - alpha * applied[index]
                residual[index] = value
                ops.atomic_add(
                    float_state[RR_NEXT * batch_size + env], value * value
                )


@kernel
def finish_solution_residual_compact(
    float_state: ndarray_type.ndarray(dtype=f32, ndim=1),
    int_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    counters: ndarray_type.ndarray(dtype=i32, ndim=1),
    batch_size: i32,
):
    for env in range(batch_size):
        float_state[RHO_NEXT * batch_size + env] = 0.0
        if int_state[ACTIVE * batch_size + env] != 0:
            rr_next = float_state[RR_NEXT * batch_size + env]
            int_state[ITERATIONS * batch_size + env] += 1
            ops.atomic_add(counters[EXECUTED_SYSTEM_ITERATIONS], 1)
            invalid = (rr_next != rr_next or rr_next < 0.0
                       or rr_next > 3.402823466e38)
            if invalid:
                float_state[RR_CURRENT * batch_size + env] = rr_next
                int_state[STATUS * batch_size + env] = 1
                int_state[ACTIVE * batch_size + env] = 0
                ops.atomic_sub(counters[ACTIVE_COUNT], 1)
            elif rr_next <= float_state[TOLERANCE_SQUARED * batch_size + env]:
                float_state[RR_CURRENT * batch_size + env] = rr_next
                int_state[STATUS * batch_size + env] = 2
                int_state[ACTIVE * batch_size + env] = 0
                ops.atomic_sub(counters[ACTIVE_COUNT], 1)


@kernel
def prepare_direction(
        source: ndarray_type.ndarray(dtype=f32, ndim=1),
        direction: ndarray_type.ndarray(dtype=f32, ndim=1),
        float_state: ndarray_type.ndarray(dtype=f32, ndim=1),
        int_state: ndarray_type.ndarray(dtype=i32, ndim=1),
        counters: ndarray_type.ndarray(dtype=i32, ndim=1),
        total_size: i32,
        system_size: i32,
        batch_size: i32,
        preconditioned: template(),
):
    for env in range(batch_size):
        if int_state[ACTIVE * batch_size + env] != 0:
            numerator = float_state[(RHO_NEXT if preconditioned else RR_NEXT) *
                                    batch_size + env]
            denominator = float_state[
                (RHO_CURRENT if preconditioned else RR_CURRENT) * batch_size +
                env]
            beta = numerator / denominator
            invalid = (numerator != numerator or numerator <= 0.0
                       or numerator > 3.402823466e38
                       or denominator != denominator or denominator <= 0.0
                       or denominator > 3.402823466e38 or beta != beta
                       or beta > 3.402823466e38 or beta < -3.402823466e38)
            if invalid:
                float_state[RR_CURRENT * batch_size +
                            env] = float_state[RR_NEXT * batch_size + env]
                int_state[STATUS * batch_size + env] = 1
                int_state[ACTIVE * batch_size + env] = 0
                ops.atomic_sub(counters[ACTIVE_COUNT], 1)
                beta = 0.0
            else:
                float_state[RR_CURRENT * batch_size +
                            env] = float_state[RR_NEXT * batch_size + env]
                if preconditioned:
                    float_state[RHO_CURRENT * batch_size + env] = numerator
            float_state[BETA * batch_size + env] = beta
        else:
            float_state[BETA * batch_size + env] = 0.0
    for index in range(total_size):
        env = index // system_size
        if int_state[ACTIVE * batch_size + env] != 0:
            direction[index] = (
                source[index] +
                float_state[BETA * batch_size + env] * direction[index])
        else:
            direction[index] = 0.0


@kernel
def prepare_direction_compact_coefficients(
    float_state: ndarray_type.ndarray(dtype=f32, ndim=1),
    int_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    counters: ndarray_type.ndarray(dtype=i32, ndim=1),
    batch_size: i32,
    preconditioned: template(),
):
    for env in range(batch_size):
        if int_state[ACTIVE * batch_size + env] != 0:
            numerator = float_state[(RHO_NEXT if preconditioned else RR_NEXT) *
                                    batch_size + env]
            denominator = float_state[
                (RHO_CURRENT if preconditioned else RR_CURRENT) * batch_size +
                env]
            beta = numerator / denominator
            invalid = (numerator != numerator or numerator <= 0.0
                       or numerator > 3.402823466e38
                       or denominator != denominator or denominator <= 0.0
                       or denominator > 3.402823466e38 or beta != beta
                       or beta > 3.402823466e38 or beta < -3.402823466e38)
            if invalid:
                float_state[RR_CURRENT * batch_size +
                            env] = float_state[RR_NEXT * batch_size + env]
                int_state[STATUS * batch_size + env] = 1
                int_state[ACTIVE * batch_size + env] = 0
                ops.atomic_sub(counters[ACTIVE_COUNT], 1)
                beta = 0.0
            else:
                float_state[RR_CURRENT * batch_size +
                            env] = float_state[RR_NEXT * batch_size + env]
                if preconditioned:
                    float_state[RHO_CURRENT * batch_size + env] = numerator
            float_state[BETA * batch_size + env] = beta
        else:
            float_state[BETA * batch_size + env] = 0.0


@kernel
def update_direction_compact_values(
    source: ndarray_type.ndarray(dtype=f32, ndim=1),
    direction: ndarray_type.ndarray(dtype=f32, ndim=1),
    float_state: ndarray_type.ndarray(dtype=f32, ndim=1),
    int_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    active_systems: ndarray_type.ndarray(dtype=i32, ndim=1),
    extent_state: ndarray_type.ndarray(dtype=i32, ndim=1),
    total_size: template(),
    system_size: i32,
    batch_size: i32,
):
    for compact_index in range(total_size):
        if compact_index < device_extent_count(extent_state):
            active_slot = compact_index // system_size
            local_index = compact_index - active_slot * system_size
            env = active_systems[active_slot]
            index = env * system_size + local_index
            if int_state[ACTIVE * batch_size + env] != 0:
                direction[index] = (
                    source[index] +
                    float_state[BETA * batch_size + env] * direction[index]
                )
            else:
                direction[index] = 0.0


@kernel
def mark_max_iterations(
        float_state: ndarray_type.ndarray(dtype=f32, ndim=1),
        int_state: ndarray_type.ndarray(dtype=i32, ndim=1),
        counters: ndarray_type.ndarray(dtype=i32, ndim=1),
        batch_size: i32,
):
    for env in range(batch_size):
        if int_state[ACTIVE * batch_size + env] != 0:
            if int_state[ITERATIONS * batch_size + env] > 0:
                float_state[RR_CURRENT * batch_size +
                            env] = float_state[RR_NEXT * batch_size + env]
            int_state[STATUS * batch_size + env] = 0
            int_state[ACTIVE * batch_size + env] = 0
    counters[ACTIVE_COUNT] = 0
