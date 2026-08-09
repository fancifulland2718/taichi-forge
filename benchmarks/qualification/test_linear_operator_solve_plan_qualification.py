from __future__ import annotations

import pytest

from benchmarks.qualification.linear_operator_solve_plan_qualification import (
    MODES,
    balanced_mode_orders,
    expected_route,
)


def test_mode_orders_alternate_without_changing_membership() -> None:
    orders = balanced_mode_orders(5)
    assert orders[0] == MODES
    assert orders[1] == tuple(reversed(MODES))
    assert all(set(order) == set(MODES) for order in orders)
    assert abs(sum(order[0] == MODES[0] for order in orders) -
               sum(order[0] == MODES[1] for order in orders)) == 1


def test_mode_orders_reject_empty_measurement() -> None:
    with pytest.raises(ValueError, match="positive"):
        balanced_mode_orders(0)


def test_backend_route_contracts_are_explicit() -> None:
    assert expected_route("cuda")["primitive"] == "cuda_conditional_graph"
    assert not expected_route("cuda")["automatic_selection_qualified"]
    assert expected_route("vulkan")["primitive"] == "vulkan_dispatch_indirect"
    assert expected_route("vulkan")["automatic_selection_qualified"]
    with pytest.raises(ValueError, match="unsupported"):
        expected_route("cpu")
