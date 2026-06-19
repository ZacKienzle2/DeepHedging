"""Tests for transaction cost models."""

import pytest
import torch

from deephedging.frictions import BidAskCost, PowerLawImpactCost, ProportionalCost


def test_impact_reduces_to_proportional_at_unit_exponent() -> None:
    trade = torch.tensor([-2.0, 1.0, 3.0], dtype=torch.float64)
    price = torch.full((3,), 100.0, dtype=torch.float64)
    impact = PowerLawImpactCost(coefficient=1e-3, exponent=1.0)(trade, price)
    proportional = ProportionalCost(rate=1e-3)(trade, price)
    assert torch.allclose(impact, proportional)


def test_impact_is_superlinear_in_trade_size() -> None:
    price = torch.tensor(100.0, dtype=torch.float64)
    cost = PowerLawImpactCost(coefficient=1e-3)
    small = float(cost(torch.tensor(1.0, dtype=torch.float64), price))
    large = float(cost(torch.tensor(2.0, dtype=torch.float64), price))
    assert large > 2.0 * small


def test_impact_is_nonnegative_and_zero_at_no_trade() -> None:
    trade = torch.tensor([-5.0, 0.0, 4.0], dtype=torch.float64)
    price = torch.full((3,), 50.0, dtype=torch.float64)
    cost = PowerLawImpactCost(coefficient=2e-3)(trade, price)
    assert bool(torch.all(cost >= 0.0))
    assert float(cost[1]) == 0.0


def test_impact_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        PowerLawImpactCost(coefficient=-1e-3)
    with pytest.raises(ValueError):
        PowerLawImpactCost(coefficient=1e-3, exponent=0.5)


def test_bid_ask_charges_each_side_its_own_half_spread() -> None:
    price = torch.tensor(100.0, dtype=torch.float64)
    cost = BidAskCost(bid_half_spread=1e-3, ask_half_spread=3e-3)
    buy = float(cost(torch.tensor(2.0, dtype=torch.float64), price))
    sell = float(cost(torch.tensor(-2.0, dtype=torch.float64), price))
    assert abs(buy - 3e-3 * 100.0 * 2.0) < 1e-12
    assert abs(sell - 1e-3 * 100.0 * 2.0) < 1e-12
    assert buy > sell


def test_bid_ask_reduces_to_proportional_when_symmetric() -> None:
    trade = torch.tensor([-2.0, 1.0, 3.0], dtype=torch.float64)
    price = torch.full((3,), 100.0, dtype=torch.float64)
    symmetric = BidAskCost(bid_half_spread=5e-4, ask_half_spread=5e-4)(trade, price)
    proportional = ProportionalCost(rate=5e-4)(trade, price)
    assert torch.allclose(symmetric, proportional)


def test_bid_ask_rejects_negative_half_spread() -> None:
    with pytest.raises(ValueError):
        BidAskCost(bid_half_spread=-1e-3, ask_half_spread=1e-3)
