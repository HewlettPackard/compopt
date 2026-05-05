"""
compopt.physics
===============

Thermo-fluid physics engine for GPU data-center simulation.

Sub-modules
-----------
chip
    GPU chip-level RC-network thermal model (``GPUChipModel``, ``make_gpu``).
electricity
    Realistic grid electricity price model (``GridElectricityPriceModel``,
    ``TOUTariff``).
fluids
    Coolant loops, CDU, cooling tower, and vectorised variants.
server
    Server, rack, and data-center aggregation models.
workloads
    Synthetic power-profile generators for benchmarking.
"""

from compopt.physics.chip import GPUChipModel, PowerProfile, GPU_PRESETS, make_gpu
from compopt.physics.electricity import GridElectricityPriceModel, TOUTariff
from compopt.physics.fluids import CoolantLoop, CDU, CoolingTower, BatchCoolantLoop
from compopt.physics.server import (
    ServerNode, RackModel, DataCenterModel,
    build_default_rack, build_default_datacenter,
)
from compopt.physics.workloads import (
    make_sinusoidal_profile, make_step_profile, make_burst_profile,
    make_trace_profile, make_mixed_profile, make_stochastic_profile,
)

__all__ = [
    "GPUChipModel", "PowerProfile", "GPU_PRESETS", "make_gpu",
    "GridElectricityPriceModel", "TOUTariff",
    "CoolantLoop", "CDU", "CoolingTower", "BatchCoolantLoop",
    "ServerNode", "RackModel", "DataCenterModel",
    "build_default_rack", "build_default_datacenter",
    "make_sinusoidal_profile", "make_step_profile", "make_burst_profile",
    "make_trace_profile", "make_mixed_profile", "make_stochastic_profile",
]
