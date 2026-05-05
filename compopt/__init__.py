"""
CompOpt — AI-Enabled Data Center Simulation Benchmark
=====================================================

An end-to-end simulator for workload scheduling, compute budgeting,
and energy / water / cost-efficient liquid-cooling control in GPU data centers.

Developed by adapting concepts from Anonymous,
RAPS (Resource Allocator & Power Simulator)
framework into a pure-Python, Gymnasium-compatible benchmark for
reinforcement learning and agentic-AI research.

Modules
-------
- compopt.physics        : Thermo-fluid models (chip, server, rack, CDU, cooling tower)
- compopt.scheduling     : Job scheduling, workload generation, RAPS-style resource allocation
- compopt.envs           : Gymnasium environments at multiple abstraction levels
- compopt.rewards        : Modular, composable reward functions
- compopt.agents         : Baseline agents (rule-based, RL wrappers, LLM/agentic)
- compopt.utils          : Logging, metrics, vectorized helpers, visualization
- compopt.configs        : YAML/JSON system configuration presets

Quick Start
-----------
>>> import compopt
>>> env = compopt.make("DataCenter-v0")           # full datacenter env
>>> env = compopt.make("RackCooling-v0")           # single-rack cooling
>>> env = compopt.make("ChipThermal-v0")           # chip-level thermal
>>> env = compopt.make("Scheduling-v0")            # job scheduling only
>>> obs, info = env.reset()
>>> action = env.action_space.sample()
>>> obs, reward, terminated, truncated, info = env.step(action)
"""

__version__ = "0.1.0"
__author__  = "anonymous"

from compopt.envs.registry import make, list_envs

__all__ = ["make", "list_envs", "__version__"]
