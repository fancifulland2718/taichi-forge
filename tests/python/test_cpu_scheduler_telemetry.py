import taichi_forge as ti
from tests import test_utils


@test_utils.test(arch=ti.cpu, cpu_max_num_threads=4, offline_cache=False)
def test_cpu_scheduler_telemetry_is_explicit_and_resettable():
    size = 4096
    values = ti.ndarray(ti.i32, shape=size)

    @ti.kernel
    def fill(output: ti.types.ndarray(dtype=ti.i32, ndim=1)):
        for i in output:
            output[i] = i

    initial = ti.profiler.query_cpu_scheduler_telemetry(reset=True)
    assert initial["schema_version"] == 1
    assert not initial["enabled"]

    ti.profiler.set_cpu_scheduler_telemetry(True, reset=True)
    for _ in range(3):
        fill(values)
    snapshot = ti.profiler.query_cpu_scheduler_telemetry()

    assert snapshot["enabled"]
    assert snapshot["jobs_submitted"] >= 3
    assert snapshot["jobs_completed"] == snapshot["jobs_submitted"]
    assert snapshot["tasks_requested"] >= snapshot["jobs_submitted"]
    assert snapshot["tasks_completed"] == snapshot["tasks_requested"]
    assert snapshot["joined_workers"] >= snapshot["jobs_submitted"]
    assert snapshot["max_requested_threads"] <= 4
    assert snapshot["max_joined_workers"] <= 4
    assert snapshot["max_queue_depth"] >= 1
    assert snapshot["execution_ns"] > 0
    assert snapshot["submitter_wait_ns"] >= snapshot["execution_ns"]
    assert snapshot["cancelled_jobs"] == 0
    assert snapshot["exception_jobs"] == 0

    retired = ti.profiler.query_cpu_scheduler_telemetry(reset=True)
    assert retired["jobs_submitted"] == snapshot["jobs_submitted"]
    cleared = ti.profiler.query_cpu_scheduler_telemetry()
    assert cleared["enabled"]
    assert cleared["jobs_submitted"] == 0
    assert cleared["tasks_requested"] == 0

    ti.profiler.set_cpu_scheduler_telemetry(False)
    fill(values)
    disabled = ti.profiler.query_cpu_scheduler_telemetry()
    assert not disabled["enabled"]
    assert disabled["jobs_submitted"] == 0
