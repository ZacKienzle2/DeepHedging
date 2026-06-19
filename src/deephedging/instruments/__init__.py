"""Derivative payoffs."""

from deephedging.instruments.asian import AsianCall, AsianPut
from deephedging.instruments.barrier import (
    DoubleKnockOutCall,
    DownAndInCall,
    DownAndOutCall,
    UpAndInCall,
    UpAndOutCall,
)
from deephedging.instruments.base import Payoff
from deephedging.instruments.basket import BasketCall, GeometricBasketCall
from deephedging.instruments.lookback import LookbackCall, LookbackPut
from deephedging.instruments.multi_asset import SingleAssetPayoff
from deephedging.instruments.vanilla import EuropeanCall, EuropeanPut

__all__ = [
    "AsianCall",
    "AsianPut",
    "BasketCall",
    "DoubleKnockOutCall",
    "DownAndInCall",
    "DownAndOutCall",
    "EuropeanCall",
    "EuropeanPut",
    "GeometricBasketCall",
    "LookbackCall",
    "LookbackPut",
    "Payoff",
    "SingleAssetPayoff",
    "UpAndInCall",
    "UpAndOutCall",
]
