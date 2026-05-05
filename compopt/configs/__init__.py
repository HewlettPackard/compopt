"""
compopt.configs
===============

Configuration system for CompOpt thermal simulation framework.

This module provides:

1. **JSON configuration presets** for reproducible data-center setups
2. **Component configurations** for GPU, coolant, CDU, and cooling tower parameters
3. **Configuration validation** to ensure numerical stability and thermal feasibility

Quick Start
-----------
>>> from compopt.configs import validate_chip_env_config, validate_system_config
>>> result = validate_chip_env_config(gpu_preset="H100_SXM", dt=0.5)
>>> if result.is_valid:
...     print("Configuration is valid!")

Available datacenter presets: ``small_ai_cluster``, ``frontier_like``,
``large_b200_cluster``.

See ``compopt/configs/components/README.md`` for detailed documentation
on parameter sizing and troubleshooting.
"""

from compopt.configs.loader import (
    list_presets as list_datacenter_presets,
    load_config,
    build_from_config,
)
from compopt.configs.validator import (
    validate_chip_env_config,
    validate_system_config,
    validate_chip_thermal_feasibility,
    validate_chip_numerical_stability,
    validate_cdu_sizing,
    validate_cooling_tower_sizing,
    load_component_config,
    list_presets as list_component_presets,
    ValidationResult,
    ValidationIssue,
    Severity,
)

__all__ = [
    # Datacenter presets
    "list_datacenter_presets",
    "load_config",
    "build_from_config",
    # Component configuration
    "load_component_config",
    "list_component_presets",
    # Validation
    "validate_chip_env_config",
    "validate_system_config",
    "validate_chip_thermal_feasibility",
    "validate_chip_numerical_stability",
    "validate_cdu_sizing",
    "validate_cooling_tower_sizing",
    "ValidationResult",
    "ValidationIssue",
    "Severity",
]
