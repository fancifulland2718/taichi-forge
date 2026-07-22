import json

import numpy as np
import pytest

import taichi_forge as ti
from tests import test_utils


def _vector(values):
    values = np.asarray(values, dtype=np.float32)
    result = ti.ndarray(ti.f32, shape=values.size)
    result.from_numpy(values)
    return result


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_operator_qualification_report_is_versioned_and_detached():
    experimental = ti.linalg.experimental
    operator = 2.0 * experimental.identity(5)
    reference = 2.0 * np.eye(5, dtype=np.float32)

    report = experimental.qualify_operator(
        operator,
        reference=reference,
        samples=2,
        warmup=0,
        repetitions=2,
        metadata={"workload": "canonical_identity"},
    )

    assert report.passed
    record = report.to_dict()
    assert record["schema"] == "taichi_forge.linalg.operator_qualification.v1"
    assert record["schema_version"] == 1
    assert record["operator"]["shape"] == [5, 5]
    assert record["operator"]["capabilities"]["adjoint_apply"]
    statuses = {check["name"]: check["status"] for check in record["checks"]}
    assert statuses["linearity"] == "passed"
    assert statuses["forward_reference"] == "passed"
    assert statuses["adjoint_dot_product"] == "passed"
    assert statuses["adjoint_reference"] == "passed"
    assert record["timing"]["boundary"] == "synchronous_public_apply"
    assert record["timing"]["warm_apply_ms"]["minimum"] >= 0.0
    assert record["metadata"]["workload"] == "canonical_identity"
    assert json.loads(report.to_json())["passed"]

    record["operator"]["shape"][0] = 99
    assert report.to_dict()["operator"]["shape"] == [5, 5]
    with pytest.raises(TypeError):
        report.record["operator"]["provider"] = "mutated"


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_operator_qualification_callable_reference_and_unsupported_adjoint():
    experimental = ti.linalg.experimental
    size = 4
    topology = ti.ndarray(ti.i32, shape=size)
    numeric = _vector([2.0, 3.0, 5.0, 7.0])
    topology.from_numpy(np.arange(size, dtype=np.int32))

    @ti.kernel
    def diagonal(
        active_size: ti.i32,
        topology_data: ti.types.ndarray(dtype=ti.i32, ndim=1),
        numeric_data: ti.types.ndarray(dtype=ti.f32, ndim=1),
        x: ti.types.ndarray(dtype=ti.f32, ndim=1),
        y: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for index in range(active_size):
            y[index] = numeric_data[index] * x[topology_data[index]]

    operator = experimental.LinearOperator.from_kernel(
        diagonal, size, topology, numeric=numeric
    )
    diagonal_host = numeric.to_numpy()
    report = experimental.qualify_operator(
        operator,
        reference=lambda values: diagonal_host * values,
        samples=2,
        repetitions=1,
    )

    assert report.passed
    statuses = {
        check["name"]: check["status"]
        for check in report.to_dict()["checks"]
    }
    assert statuses["forward_reference"] == "passed"
    assert statuses["adjoint_dot_product"] == "unsupported"


@test_utils.test(arch=ti.cpu, offline_cache=False)
def test_operator_qualification_rejects_invalid_controls_and_oracle_shape():
    experimental = ti.linalg.experimental
    operator = experimental.identity(3)

    with pytest.raises(RuntimeError, match="samples must be positive"):
        experimental.qualify_operator(operator, samples=0)
    with pytest.raises(RuntimeError, match="reference must have shape"):
        experimental.qualify_operator(
            operator, reference=np.eye(4, dtype=np.float32)
        )
    with pytest.raises(RuntimeError, match="JSON-serializable"):
        experimental.qualify_operator(operator, metadata={"bad": object()})
