"""Conditional flow matching (Lipman et al., 2023) on 2-D data, from scratch."""

from .data import make_two_moons_labelled
from .flow import generate, interpolate, mean_nn_distance, train
from .model import SinusoidalTime, VelocityField

__version__ = "0.2.0"

__all__ = [
    "SinusoidalTime",
    "VelocityField",
    "generate",
    "interpolate",
    "make_two_moons_labelled",
    "mean_nn_distance",
    "train",
]
