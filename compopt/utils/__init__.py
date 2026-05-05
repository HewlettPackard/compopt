"""
compopt.utils
=============

Utilities: vectorised simulation, evaluation metrics, and plotting.

Sub-modules
-----------
vec_env
    ``BatchSimulator`` for parallel environment stepping.
metrics
    ``EpisodeMetrics``, ``evaluate_agent``, ``print_metrics``.
plotting
    Publication-quality figures for NeurIPS papers.
"""

from compopt.utils.vec_env import BatchSimulator, benchmark_throughput
from compopt.utils.metrics import EpisodeMetrics, evaluate_agent, print_metrics

__all__ = [
    "BatchSimulator", "benchmark_throughput",
    "EpisodeMetrics", "evaluate_agent", "print_metrics",
]
