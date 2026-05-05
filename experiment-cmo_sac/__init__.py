"""
CMO-SAC: Constrained Multi-Objective Soft Actor-Critic

A constrained multi-objective extension of SAC for data center cooling
that handles conflicting objectives (PUE, WUE, throughput) with hard
constraints (thermal limits, SLA).

Key features:
- CMDP formulation with vector rewards
- Lagrangian dual decomposition for constraint handling
- Pareto front generation via weight sweeping
- Safety filter with one-step thermal prediction
"""

__version__ = "0.1.0"
