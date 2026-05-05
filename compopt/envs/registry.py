"""
compopt.envs.registry
=====================
Unified environment registry — ``compopt.make("EnvName-v0")`` interface.

Supports all four difficulty levels:
    - ChipThermal-v0    : single GPU chip thermal control (Easy)
    - RackCooling-v0    : rack-level cooling control (Medium)
    - Scheduling-v0     : job scheduling only (Medium)
    - DataCenter-v0     : full datacenter cooling (Hard)
    - JointDC-v0        : joint scheduling + cooling (Expert)
    - JointDCFlat-v0    : same as JointDC-v0 with flat obs/action (Expert)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

# Lazy imports to avoid heavy dependencies at import time
_REGISTRY: Dict[str, Dict[str, Any]] = {
    "ChipThermal-v0": {
        "entry_point": "compopt.envs.chip_env:ChipThermalEnv",
        "kwargs": {},
        "description": "Single GPU chip thermal control (Easy)",
    },
    "RackCooling-v0": {
        "entry_point": "compopt.envs.rack_env:RackCoolingEnv",
        "kwargs": {},
        "description": "Rack-level liquid-cooling control (Medium)",
    },
    "Scheduling-v0": {
        "entry_point": "compopt.envs.scheduling_env:SchedulingEnv",
        "kwargs": {},
        "description": "Data-center job scheduling (Medium)",
    },
    "DataCenter-v0": {
        "entry_point": "compopt.envs.datacenter_env:DataCenterEnv",
        "kwargs": {},
        "description": "Full data-center cooling control (Hard)",
    },
    "JointDC-v0": {
        "entry_point": "compopt.envs.joint_env:JointDataCenterEnv",
        "kwargs": {"flatten": False},
        "description": "Joint scheduling + cooling with Dict spaces (Expert)",
    },
    "JointDCFlat-v0": {
        "entry_point": "compopt.envs.joint_env:JointDataCenterEnv",
        "kwargs": {"flatten": True},
        "description": "Joint scheduling + cooling, flattened spaces (Expert)",
    },
}


def _import_class(entry_point: str):
    """Dynamically import a class from a module:class string."""
    module_path, class_name = entry_point.rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def make(env_id: str, normalize_obs: bool = True, normalize_reward: bool = False, **kwargs) -> Any:
    """
    Create a CompOpt Gymnasium environment by name.
    
    By default, applies observation normalization to prevent numerical
    instability from large observation values. This can be disabled by
    setting normalize_obs=False.

    Parameters
    ----------
    env_id : str
        One of the registered environment IDs (see ``list_envs()``).
    normalize_obs : bool, default=True
        Whether to apply observation normalization wrapper.
        Recommended to keep enabled for neural network training.
    normalize_reward : bool, default=False
        Whether to apply reward normalization wrapper.
        Generally not needed for CompOpt environments.
    **kwargs
        Override any default environment parameters.

    Returns
    -------
    gymnasium.Env instance

    Examples
    --------
    >>> import compopt
    >>> env = compopt.make("RackCooling-v0", dt=2.0, episode_length_s=900)
    >>> obs, info = env.reset()
    
    >>> # Disable normalization if needed (e.g., for rule-based agents)
    >>> env = compopt.make("ChipThermal-v0", normalize_obs=False)
    """
    if env_id not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown env_id '{env_id}'. Available: {available}")

    spec = _REGISTRY[env_id]
    cls  = _import_class(spec["entry_point"])
    merged_kwargs = {**spec["kwargs"], **kwargs}
    env = cls(**merged_kwargs)
    
    # Apply wrappers for numerical stability
    if normalize_obs or normalize_reward:
        from compopt.envs.wrappers import apply_wrappers
        env = apply_wrappers(
            env, 
            normalize_obs=normalize_obs,
            normalize_reward=normalize_reward
        )
    
    return env


def list_envs() -> List[Dict[str, str]]:
    """List all registered CompOpt environments."""
    return [
        {"id": eid, "description": spec["description"]}
        for eid, spec in _REGISTRY.items()
    ]


def register(env_id: str, entry_point: str,
             description: str = "", **kwargs):
    """Register a custom environment."""
    _REGISTRY[env_id] = {
        "entry_point": entry_point,
        "kwargs": kwargs,
        "description": description,
    }
