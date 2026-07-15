"""Optimized Hamiltonian-learning experiments."""

from .models import ExperimentModel, one_qubit_model, two_qubit_model
from .rfe import (
    ProbabilityTable,
    RFESchedule,
    build_probability_table,
    build_rfe_schedule,
    simulate_gap_estimates,
)
from .solver import FitResult, fit_gap_targets, gaps_and_jacobian

__all__ = [
    "ExperimentModel",
    "FitResult",
    "ProbabilityTable",
    "RFESchedule",
    "build_probability_table",
    "build_rfe_schedule",
    "fit_gap_targets",
    "gaps_and_jacobian",
    "one_qubit_model",
    "simulate_gap_estimates",
    "two_qubit_model",
]