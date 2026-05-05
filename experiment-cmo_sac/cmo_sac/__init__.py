"""
CMO-SAC: Constrained Multi-Objective Soft Actor-Critic

This package implements the CMO-SAC algorithm for data center cooling
with multi-objective optimization and constraint satisfaction.
"""

__version__ = "0.1.0"

# Lazy imports to handle missing dependencies gracefully
def __getattr__(name):
    """Lazy import of submodules."""
    if name == "MultiObjectiveReward":
        from .rewards import MultiObjectiveReward
        return MultiObjectiveReward
    elif name == "ConstraintManager":
        from .constraints import ConstraintManager
        return ConstraintManager
    elif name == "LagrangianDualOptimizer":
        from .constraints import LagrangianDualOptimizer
        return LagrangianDualOptimizer
    elif name == "CMOCritic":
        from .networks import CMOCritic
        return CMOCritic
    elif name == "CMOActor":
        from .networks import CMOActor
        return CMOActor
    elif name == "CMOSACAgent":
        from .cmo_sac_agent import CMOSACAgent
        return CMOSACAgent
    elif name == "CMOSACTrainer":
        from .trainer import CMOSACTrainer
        return CMOSACTrainer
    elif name == "ParetoFrontGenerator":
        from .pareto import ParetoFrontGenerator
        return ParetoFrontGenerator
    elif name == "SafetyFilter":
        from .safety import SafetyFilter
        return SafetyFilter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MultiObjectiveReward",
    "ConstraintManager", 
    "LagrangianDualOptimizer",
    "CMOCritic",
    "CMOActor",
    "CMOSACAgent",
    "CMOSACTrainer",
    "ParetoFrontGenerator",
    "SafetyFilter",
]
