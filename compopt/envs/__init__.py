"""
compopt.envs
============

Gymnasium-compatible RL environments at multiple abstraction levels.

Environments
------------
ChipThermalEnv
    Single GPU chip thermal control (**Easy**).
RackCoolingEnv
    Rack-level liquid-cooling control (**Medium**).
SchedulingEnv
    Data-center job scheduling (**Medium**).
DataCenterEnv
    Full data-center cooling control (**Hard**).
JointDataCenterEnv
    Joint scheduling + cooling (**Expert**).

All environments can be created via :func:`compopt.make`:

>>> import compopt
>>> env = compopt.make("RackCooling-v0")
"""

from compopt.envs.chip_env import ChipThermalEnv
from compopt.envs.rack_env import RackCoolingEnv
from compopt.envs.datacenter_env import DataCenterEnv
from compopt.envs.scheduling_env import SchedulingEnv
from compopt.envs.joint_env import JointDataCenterEnv
from compopt.envs.registry import make, list_envs, register

__all__ = [
    "ChipThermalEnv", "RackCoolingEnv", "DataCenterEnv",
    "SchedulingEnv", "JointDataCenterEnv",
    "make", "list_envs", "register",
]
