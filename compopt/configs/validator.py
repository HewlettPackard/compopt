"""
compopt.configs.validator
=========================
Configuration validation for CompOpt physical parameters.

This module provides validation functions to ensure that user-specified
configurations are physically feasible and numerically stable. It checks:

1. **Thermal feasibility**: Can the cooling system handle the heat load?
2. **Numerical stability**: Are timesteps small enough for stable integration?
3. **Control authority**: Does the controller have enough range to regulate temperature?
4. **Parameter consistency**: Are all related parameters properly scaled?

Usage
-----
>>> from compopt.configs.validator import validate_chip_config, validate_system_config
>>> config = load_component_config("gpu_presets", "H100_SXM")
>>> issues = validate_chip_config(config, coolant_config, simulation_config)
>>> if issues:
...     for issue in issues:
...         print(f"{issue.severity}: {issue.message}")
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class Severity(Enum):
    """Severity level of validation issues."""
    INFO = "INFO"           # Informational, no action needed
    WARNING = "WARNING"     # May cause suboptimal behavior
    ERROR = "ERROR"         # Will likely cause simulation issues
    CRITICAL = "CRITICAL"   # Will definitely fail or produce invalid results


@dataclass
class ValidationIssue:
    """A single validation issue found in the configuration."""
    severity: Severity
    category: str
    message: str
    parameter: Optional[str] = None
    suggestion: Optional[str] = None
    
    def __str__(self) -> str:
        s = f"[{self.severity.value}] {self.category}: {self.message}"
        if self.parameter:
            s += f" (parameter: {self.parameter})"
        if self.suggestion:
            s += f"\n  → Suggestion: {self.suggestion}"
        return s


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    issues: List[ValidationIssue]
    
    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        lines = [f"Configuration validation: {status}"]
        for issue in self.issues:
            lines.append(str(issue))
        return "\n".join(lines)
    
    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity in (Severity.ERROR, Severity.CRITICAL)]
    
    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]


def _get_config_dir() -> Path:
    """Get the path to the configs/components directory."""
    return Path(__file__).parent / "components"


def load_component_config(component: str, preset: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a component configuration from JSON.
    
    Parameters
    ----------
    component : str
        Component name: "gpu_presets", "coolant_loops", "cdu", "cooling_tower", 
        "simulation", or "workloads"
    preset : str, optional
        Specific preset within the component (e.g., "H100_SXM" for gpu_presets)
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    config_path = _get_config_dir() / f"{component}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    if preset is not None:
        if preset not in config:
            available = [k for k in config.keys() if not k.startswith("_")]
            raise KeyError(f"Preset '{preset}' not found in {component}. "
                          f"Available: {available}")
        return config[preset]
    
    return config


def list_presets(component: str) -> List[str]:
    """List available presets for a component."""
    config = load_component_config(component)
    return [k for k in config.keys() if not k.startswith("_")]


# ══════════════════════════════════════════════════════════════════════════════
# Chip-Level Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_chip_thermal_feasibility(
    gpu_config: Dict[str, Any],
    coolant_config: Dict[str, Any],
    target_temp_C: float = 75.0
) -> List[ValidationIssue]:
    """
    Validate that the cooling system can maintain GPU within thermal limits.
    
    Checks:
    1. At max flow + TDP, temperature should be below target
    2. At min flow + TDP, temperature should be near (but below) T_max
    3. Temperature range should be meaningful for control
    """
    issues = []
    
    # Extract parameters
    tdp_W = gpu_config["tdp_W"]
    n_tiles = gpu_config["nx_tiles"] * gpu_config["ny_tiles"]
    R_tile = gpu_config["R_tile_to_coolant_K_W"]
    T_max = gpu_config["T_max_die_C"]
    
    T_in = coolant_config["T_in_C"]
    m_dot_min = coolant_config["m_dot_min_kg_s"]
    m_dot_max = coolant_config["m_dot_max_kg_s"]
    c_p = coolant_config["c_p_J_kgK"]
    
    # Effective thermal resistance (tiles in parallel)
    R_eff = R_tile / n_tiles
    
    # Check for zero/negative flow rates
    if m_dot_min <= 0:
        issues.append(ValidationIssue(
            severity=Severity.ERROR,
            category="Invalid Parameter",
            message=f"m_dot_min = {m_dot_min} kg/s is invalid (must be > 0)",
            parameter="m_dot_min_kg_s",
            suggestion="Set m_dot_min to a positive value (e.g., 0.01 kg/s)"
        ))
        m_dot_min = 0.01  # Use fallback for remaining calculations
    
    if m_dot_max <= 0:
        issues.append(ValidationIssue(
            severity=Severity.ERROR,
            category="Invalid Parameter",
            message=f"m_dot_max = {m_dot_max} kg/s is invalid (must be > 0)",
            parameter="m_dot_max_kg_s",
            suggestion="Set m_dot_max to a positive value (e.g., 0.5 kg/s)"
        ))
        m_dot_max = 0.5  # Use fallback for remaining calculations
    
    # Steady-state temperature calculations
    # At max flow: T_die = T_coolant + R_eff * P
    # T_coolant = T_in + P / (m_dot * c_p)
    
    def estimate_T_die(m_dot: float, P: float) -> float:
        if m_dot <= 0 or c_p <= 0:
            return float('inf')
        T_coolant = T_in + P / (m_dot * c_p)
        T_die = T_coolant + R_eff * P
        return T_die
    
    T_die_at_max_flow = estimate_T_die(m_dot_max, tdp_W)
    T_die_at_min_flow = estimate_T_die(m_dot_min, tdp_W)
    
    # Check 1: Temperature at max flow should be reasonable
    if T_die_at_max_flow > target_temp_C:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            category="Thermal Feasibility",
            message=f"At max flow ({m_dot_max} kg/s), T_die = {T_die_at_max_flow:.1f}°C "
                    f"exceeds target {target_temp_C}°C",
            parameter="m_dot_max_kg_s",
            suggestion=f"Increase m_dot_max or decrease R_tile_to_coolant_K_W. "
                       f"Current R_eff = {R_eff:.4f} K/W"
        ))
    
    if T_die_at_max_flow > T_max:
        issues.append(ValidationIssue(
            severity=Severity.ERROR,
            category="Thermal Feasibility",
            message=f"At max flow + TDP, T_die = {T_die_at_max_flow:.1f}°C "
                    f"exceeds thermal limit {T_max}°C. System CANNOT be cooled!",
            parameter="R_tile_to_coolant_K_W",
            suggestion="Significantly reduce thermal resistance or increase cooling capacity"
        ))
    
    # Check 2: Temperature at min flow should be high but not exceeding limit
    if T_die_at_min_flow > T_max:
        headroom = T_max - T_die_at_max_flow
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            category="Control Range",
            message=f"At min flow ({m_dot_min} kg/s) + TDP, T_die = {T_die_at_min_flow:.1f}°C "
                    f"exceeds thermal limit {T_max}°C",
            suggestion=f"Increase m_dot_min or improve cooling. "
                       f"Headroom at max flow: {headroom:.1f}°C"
        ))
    
    # Check 3: Control authority (temperature range)
    T_range = T_die_at_min_flow - T_die_at_max_flow
    if T_range < 5.0:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            category="Control Authority",
            message=f"Temperature range between min/max flow is only {T_range:.1f}°C. "
                    f"Controller may have insufficient authority.",
            suggestion="Increase flow rate range (m_dot_max/m_dot_min ratio) or "
                      "decrease coolant c_p for larger temperature swings"
        ))
    elif T_range < 10.0:
        issues.append(ValidationIssue(
            severity=Severity.INFO,
            category="Control Authority",
            message=f"Temperature range between min/max flow is {T_range:.1f}°C. "
                    f"Adequate for basic control."
        ))
    
    return issues


def validate_chip_numerical_stability(
    gpu_config: Dict[str, Any],
    coolant_config: Dict[str, Any],
    dt: float
) -> List[ValidationIssue]:
    """
    Validate numerical stability of chip thermal simulation.
    
    Checks RK4 stability criterion for the thermal RC network.
    """
    issues = []
    
    # Estimate smallest time constant
    C_min = min(gpu_config["C_tile_J_K"], gpu_config["C_hbm_J_K"], 
                gpu_config["C_vrm_J_K"])
    R_min = min(gpu_config["R_tile_to_coolant_K_W"], 
                gpu_config["R_hbm_to_coolant_K_W"],
                gpu_config["R_vrm_to_coolant_K_W"])
    
    # Conductance = 1/R, time constant τ = C / G = C * R
    # But for coolant coupling: G_cool = 1/R_to_coolant
    # Effective eigenvalue ≈ G_cool / C
    
    G_max = 1.0 / R_min
    tau_min = C_min / G_max  # Fastest time constant
    
    # RK4 stability: dt < 2.78 * tau for stability on negative real axis
    max_stable_dt = 2.78 * tau_min * 0.9  # 10% safety margin
    
    if dt > max_stable_dt:
        issues.append(ValidationIssue(
            severity=Severity.ERROR,
            category="Numerical Stability",
            message=f"Timestep dt={dt}s exceeds stability limit {max_stable_dt:.2f}s. "
                    f"Simulation may blow up!",
            parameter="dt",
            suggestion=f"Reduce dt to at most {max_stable_dt:.2f}s, or increase "
                       f"thermal capacitances (C values)"
        ))
    elif dt > max_stable_dt * 0.5:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            category="Numerical Stability",
            message=f"Timestep dt={dt}s is close to stability limit {max_stable_dt:.2f}s. "
                    f"Consider reducing for safety.",
            parameter="dt"
        ))
    else:
        issues.append(ValidationIssue(
            severity=Severity.INFO,
            category="Numerical Stability",
            message=f"Timestep dt={dt}s is safely below limit {max_stable_dt:.2f}s. "
                    f"Stability margin: {(max_stable_dt - dt) / max_stable_dt * 100:.0f}%"
        ))
    
    return issues


# ══════════════════════════════════════════════════════════════════════════════
# CDU and Cooling Tower Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_cdu_sizing(
    cdu_config: Dict[str, Any],
    expected_heat_load_W: float,
    dt: float
) -> List[ValidationIssue]:
    """
    Validate CDU heat exchanger sizing and stability.
    """
    issues = []
    
    UA = cdu_config["UA_W_K"]
    V_hot = cdu_config.get("hot_loop_V_m3", 0.05)
    V_cold = cdu_config.get("cold_loop_V_m3", 0.10)
    
    # Thermal capacitance (assume water: ρ=1000, c_p=4000)
    C_hot = V_hot * 1000 * 4000
    C_cold = V_cold * 1000 * 4000
    C_min = min(C_hot, C_cold)
    
    # Check UA vs thermal mass ratio for stability
    # Stability criterion: UA * dt / C < 0.5
    stability_ratio = UA * dt / C_min
    
    if stability_ratio > 0.5:
        issues.append(ValidationIssue(
            severity=Severity.ERROR,
            category="CDU Stability",
            message=f"UA×dt/C = {stability_ratio:.2f} > 0.5. "
                    f"CDU temperatures will oscillate wildly!",
            parameter="UA_W_K",
            suggestion=f"Reduce UA to {0.5 * C_min / dt:.0f} W/K, "
                       f"or increase loop volumes (V_m3)"
        ))
    elif stability_ratio > 0.3:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            category="CDU Stability",
            message=f"UA×dt/C = {stability_ratio:.2f} is marginal. "
                    f"May see some temperature oscillation.",
            suggestion="Consider increasing loop thermal mass"
        ))
    
    # Check UA sizing for heat load
    # Q = UA * ΔT, so ΔT = Q / UA
    delta_T = expected_heat_load_W / UA
    
    if delta_T > 20:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            category="CDU Sizing",
            message=f"Expected temperature difference across HX: {delta_T:.1f}°C. "
                    f"May be too high for efficient operation.",
            suggestion=f"Increase UA_W_K for better heat transfer"
        ))
    elif delta_T < 2:
        issues.append(ValidationIssue(
            severity=Severity.INFO,
            category="CDU Sizing",
            message=f"UA is oversized for expected load. ΔT = {delta_T:.1f}°C"
        ))
    
    return issues


def validate_cooling_tower_sizing(
    tower_config: Dict[str, Any],
    expected_heat_load_W: float,
    dt: float
) -> List[ValidationIssue]:
    """
    Validate cooling tower sizing and stability.
    """
    issues = []
    
    fan_UA = tower_config["fan_UA_W_K"]
    basin_C = tower_config["basin_C_J_K"]
    T_ambient = tower_config["T_ambient_C"]
    T_wetbulb = tower_config["T_wetbulb_C"]
    evap_frac = tower_config["evap_fraction"]
    
    # Effective rejection temperature
    T_eff = evap_frac * T_wetbulb + (1 - evap_frac) * T_ambient
    
    # Steady-state basin temperature
    # Q_reject = fan_UA * (T_basin - T_eff) = expected_heat_load_W
    T_basin_steady = T_eff + expected_heat_load_W / fan_UA
    
    if T_basin_steady > 50:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            category="Cooling Tower Sizing",
            message=f"Steady-state basin temperature: {T_basin_steady:.1f}°C. "
                    f"May be too high for efficient cooling.",
            parameter="fan_UA_W_K",
            suggestion=f"Increase fan_UA_W_K for more heat rejection capacity"
        ))
    
    # Stability check: fan_UA * dt / basin_C < 0.5
    stability_ratio = fan_UA * dt / basin_C
    
    if stability_ratio > 0.5:
        issues.append(ValidationIssue(
            severity=Severity.ERROR,
            category="Cooling Tower Stability",
            message=f"fan_UA×dt/basin_C = {stability_ratio:.2f} > 0.5. "
                    f"Basin temperature will be unstable!",
            parameter="basin_C_J_K",
            suggestion=f"Increase basin_C_J_K to at least {2 * fan_UA * dt:.0f} J/K"
        ))
    
    # WUE estimate
    latent_heat = tower_config["latent_heat_J_kg"]
    water_rate_kg_s = evap_frac * expected_heat_load_W / latent_heat
    water_rate_L_hr = water_rate_kg_s * 3600
    energy_kWh_hr = expected_heat_load_W / 1000  # per hour
    wue = water_rate_L_hr / energy_kWh_hr if energy_kWh_hr > 0 else 0
    
    if wue > 5:
        issues.append(ValidationIssue(
            severity=Severity.WARNING,
            category="Water Usage",
            message=f"Estimated WUE: {wue:.1f} L/kWh. This is high.",
            suggestion="Consider reducing evap_fraction or using dry cooling"
        ))
    else:
        issues.append(ValidationIssue(
            severity=Severity.INFO,
            category="Water Usage",
            message=f"Estimated WUE: {wue:.1f} L/kWh (typical range: 1-3)"
        ))
    
    return issues


# ══════════════════════════════════════════════════════════════════════════════
# System-Level Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_system_config(
    gpu_preset: str = "H100_SXM",
    n_gpus: int = 8,
    cdu_preset: str = "small_datacenter",
    tower_preset: str = "small_tower",
    dt: float = 5.0,
    target_temp_C: float = 75.0
) -> ValidationResult:
    """
    Validate a complete datacenter system configuration.
    
    This is the main entry point for configuration validation.
    
    Parameters
    ----------
    gpu_preset : str
        GPU preset name from gpu_presets.json
    n_gpus : int
        Total number of GPUs in the system
    cdu_preset : str
        CDU preset name from cdu.json
    tower_preset : str
        Cooling tower preset from cooling_tower.json
    dt : float
        Simulation timestep in seconds
    target_temp_C : float
        Target temperature for control
        
    Returns
    -------
    ValidationResult
        Validation result with all issues found
    """
    all_issues: List[ValidationIssue] = []
    
    try:
        gpu_config = load_component_config("gpu_presets", gpu_preset)
        coolant_config = load_component_config("coolant_loops", "chip_cooling")
        cdu_config = load_component_config("cdu", cdu_preset)
        tower_config = load_component_config("cooling_tower", tower_preset)
    except (FileNotFoundError, KeyError) as e:
        return ValidationResult(
            is_valid=False,
            issues=[ValidationIssue(
                severity=Severity.CRITICAL,
                category="Configuration Loading",
                message=str(e)
            )]
        )
    
    # Total heat load estimate
    total_heat_W = n_gpus * gpu_config["tdp_W"]
    
    # Chip-level validation
    all_issues.extend(validate_chip_thermal_feasibility(
        gpu_config, coolant_config, target_temp_C))
    all_issues.extend(validate_chip_numerical_stability(
        gpu_config, coolant_config, dt))
    
    # CDU validation
    all_issues.extend(validate_cdu_sizing(cdu_config, total_heat_W, dt))
    
    # Cooling tower validation
    all_issues.extend(validate_cooling_tower_sizing(tower_config, total_heat_W, dt))
    
    # System-level checks
    all_issues.append(ValidationIssue(
        severity=Severity.INFO,
        category="System Summary",
        message=f"Total IT load: {total_heat_W/1000:.1f} kW from {n_gpus} GPUs"
    ))
    
    # Determine overall validity
    has_errors = any(i.severity in (Severity.ERROR, Severity.CRITICAL) 
                     for i in all_issues)
    
    return ValidationResult(is_valid=not has_errors, issues=all_issues)


def validate_chip_env_config(
    gpu_preset: str = "H100_SXM",
    m_dot_min: float = 0.05,
    m_dot_max: float = 0.5,
    c_p: float = 800.0,
    dt: float = 0.5,
    target_temp_C: float = 70.0
) -> ValidationResult:
    """
    Validate configuration for ChipThermal-v0 environment.
    """
    all_issues: List[ValidationIssue] = []
    
    try:
        gpu_config = load_component_config("gpu_presets", gpu_preset)
    except (FileNotFoundError, KeyError) as e:
        return ValidationResult(
            is_valid=False,
            issues=[ValidationIssue(
                severity=Severity.CRITICAL,
                category="Configuration Loading",
                message=str(e)
            )]
        )
    
    # Build coolant config from parameters
    coolant_config = {
        "T_in_C": 40.0,
        "m_dot_min_kg_s": m_dot_min,
        "m_dot_max_kg_s": m_dot_max,
        "c_p_J_kgK": c_p
    }
    
    all_issues.extend(validate_chip_thermal_feasibility(
        gpu_config, coolant_config, target_temp_C))
    all_issues.extend(validate_chip_numerical_stability(
        gpu_config, coolant_config, dt))
    
    has_errors = any(i.severity in (Severity.ERROR, Severity.CRITICAL) 
                     for i in all_issues)
    
    return ValidationResult(is_valid=not has_errors, issues=all_issues)


# ══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Command-line interface for configuration validation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate CompOpt configuration parameters")
    parser.add_argument("--gpu", default="H100_SXM", 
                       help="GPU preset name")
    parser.add_argument("--n-gpus", type=int, default=8,
                       help="Number of GPUs")
    parser.add_argument("--cdu", default="small_datacenter",
                       help="CDU preset name")
    parser.add_argument("--tower", default="small_tower",
                       help="Cooling tower preset name")
    parser.add_argument("--dt", type=float, default=5.0,
                       help="Simulation timestep (seconds)")
    parser.add_argument("--target-temp", type=float, default=75.0,
                       help="Target temperature (°C)")
    parser.add_argument("--list-presets", action="store_true",
                       help="List available presets and exit")
    
    args = parser.parse_args()
    
    if args.list_presets:
        print("Available presets:")
        for component in ["gpu_presets", "cdu", "cooling_tower"]:
            print(f"\n{component}:")
            for preset in list_presets(component):
                print(f"  - {preset}")
        return
    
    result = validate_system_config(
        gpu_preset=args.gpu,
        n_gpus=args.n_gpus,
        cdu_preset=args.cdu,
        tower_preset=args.tower,
        dt=args.dt,
        target_temp_C=args.target_temp
    )
    
    print(result)
    
    if not result.is_valid:
        exit(1)


if __name__ == "__main__":
    main()
