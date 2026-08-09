from __future__ import annotations

import pytest

from benchmarks.qualification.linear_operator_solve_plan_qualification import (
    MODES,
    balanced_mode_orders,
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
