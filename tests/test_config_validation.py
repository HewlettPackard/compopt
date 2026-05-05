"""
tests/test_config_validation.py
===============================
Tests for the CompOpt configuration validation system.

These tests ensure that:
1. Default configurations are valid and stable
2. Invalid configurations are properly detected
3. Edge cases are handled correctly
4. Users get helpful error messages
"""

import pytest
import numpy as np
from compopt.configs.validator import (
    validate_chip_env_config,
    validate_system_config,
    validate_chip_thermal_feasibility,
    validate_chip_numerical_stability,
    validate_cdu_sizing,
    validate_cooling_tower_sizing,
    load_component_config,
    list_presets,
    ValidationResult,
    ValidationIssue,
    Severity,
)


# ══════════════════════════════════════════════════════════════════════════════
# Test Configuration Loading
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigLoading:
    """Test configuration file loading."""
    
    def test_load_gpu_presets(self):
        """GPU presets should load without error."""
        config = load_component_config("gpu_presets", "H100_SXM")
        assert "tdp_W" in config
        assert config["tdp_W"] == 700
        assert config["nx_tiles"] == 4
        assert config["ny_tiles"] == 4
    
    def test_load_all_gpu_presets(self):
        """All GPU presets should be loadable."""
        presets = list_presets("gpu_presets")
        assert "H100_SXM" in presets
        assert "B200" in presets
        
        for preset in presets:
            config = load_component_config("gpu_presets", preset)
            assert "tdp_W" in config
            assert "R_tile_to_coolant_K_W" in config
    
    def test_load_coolant_loops(self):
        """Coolant loop configs should load."""
        config = load_component_config("coolant_loops", "chip_cooling")
        assert "T_in_C" in config
        assert "c_p_J_kgK" in config
    
    def test_load_cdu_presets(self):
        """CDU presets should load."""
        presets = list_presets("cdu")
        assert len(presets) >= 1
        
        config = load_component_config("cdu", presets[0])
        assert "UA_W_K" in config
    
    def test_load_cooling_tower_presets(self):
        """Cooling tower presets should load."""
        presets = list_presets("cooling_tower")
        assert len(presets) >= 1
        
        config = load_component_config("cooling_tower", presets[0])
        assert "fan_UA_W_K" in config
        assert "basin_C_J_K" in config
    
    def test_load_invalid_preset_raises(self):
        """Loading invalid preset should raise KeyError."""
        with pytest.raises(KeyError):
            load_component_config("gpu_presets", "NONEXISTENT_GPU")
    
    def test_load_invalid_component_raises(self):
        """Loading invalid component should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_component_config("nonexistent_component")


# ══════════════════════════════════════════════════════════════════════════════
# Test Default Configurations
# ══════════════════════════════════════════════════════════════════════════════

class TestDefaultConfigurations:
    """Test that default configurations are valid."""
    
    def test_default_chip_config_valid(self):
        """Default ChipThermal-v0 configuration should be valid."""
        result = validate_chip_env_config(
            gpu_preset="H100_SXM",
            m_dot_min=0.05,
            m_dot_max=0.5,
            c_p=800.0,
            dt=0.5,
            target_temp_C=70.0
        )
        assert result.is_valid, f"Default chip config invalid: {result}"
    
    def test_default_system_config_valid(self):
        """Default system configuration should be valid."""
        result = validate_system_config(
            gpu_preset="H100_SXM",
            n_gpus=8,
            cdu_preset="small_datacenter",
            tower_preset="small_tower",
            dt=5.0,
            target_temp_C=75.0
        )
        assert result.is_valid, f"Default system config invalid: {result}"
    
    def test_all_gpu_presets_with_defaults(self):
        """All GPU presets should work with default cooling parameters."""
        for preset in list_presets("gpu_presets"):
            result = validate_chip_env_config(
                gpu_preset=preset,
                m_dot_min=0.05,
                m_dot_max=0.5,
                c_p=800.0,
                dt=0.5,
                target_temp_C=70.0
            )
            # Should at least not have critical errors
            critical_errors = [i for i in result.issues 
                              if i.severity == Severity.CRITICAL]
            assert len(critical_errors) == 0, \
                f"GPU preset {preset} has critical errors: {critical_errors}"


# ══════════════════════════════════════════════════════════════════════════════
# Test Thermal Feasibility Validation
# ══════════════════════════════════════════════════════════════════════════════

class TestThermalFeasibility:
    """Test thermal feasibility validation."""
    
    def test_adequate_cooling_no_errors(self):
        """Adequate cooling should produce no errors."""
        gpu_config = {
            "tdp_W": 700,
            "nx_tiles": 4,
            "ny_tiles": 4,
            "R_tile_to_coolant_K_W": 0.8,
            "T_max_die_C": 83.0
        }
        coolant_config = {
            "T_in_C": 40.0,
            "m_dot_min_kg_s": 0.05,
            "m_dot_max_kg_s": 0.5,
            "c_p_J_kgK": 800.0
        }
        
        issues = validate_chip_thermal_feasibility(
            gpu_config, coolant_config, target_temp_C=70.0)
        
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0, f"Unexpected errors: {errors}"
    
    def test_insufficient_cooling_detected(self):
        """Insufficient cooling should be detected as error."""
        gpu_config = {
            "tdp_W": 1000,  # High power
            "nx_tiles": 4,
            "ny_tiles": 4,
            "R_tile_to_coolant_K_W": 2.0,  # High resistance
            "T_max_die_C": 83.0
        }
        coolant_config = {
            "T_in_C": 40.0,
            "m_dot_min_kg_s": 0.05,
            "m_dot_max_kg_s": 0.2,  # Low max flow
            "c_p_J_kgK": 800.0
        }
        
        issues = validate_chip_thermal_feasibility(
            gpu_config, coolant_config, target_temp_C=70.0)
        
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) > 0, "Should detect insufficient cooling"
    
    def test_low_control_authority_warning(self):
        """Low control authority should produce warning."""
        gpu_config = {
            "tdp_W": 700,
            "nx_tiles": 4,
            "ny_tiles": 4,
            "R_tile_to_coolant_K_W": 0.8,
            "T_max_die_C": 83.0
        }
        coolant_config = {
            "T_in_C": 40.0,
            "m_dot_min_kg_s": 0.45,  # Very close to max
            "m_dot_max_kg_s": 0.5,
            "c_p_J_kgK": 4000.0  # High c_p reduces temp range
        }
        
        issues = validate_chip_thermal_feasibility(
            gpu_config, coolant_config, target_temp_C=70.0)
        
        warnings = [i for i in issues 
                   if i.severity == Severity.WARNING and "authority" in i.category.lower()]
        assert len(warnings) > 0, "Should warn about low control authority"


# ══════════════════════════════════════════════════════════════════════════════
# Test Numerical Stability Validation
# ══════════════════════════════════════════════════════════════════════════════

class TestNumericalStability:
    """Test numerical stability validation."""
    
    def test_stable_timestep_ok(self):
        """Stable timestep should not produce errors."""
        gpu_config = {
            "C_tile_J_K": 8.0,
            "C_hbm_J_K": 80.0,
            "C_vrm_J_K": 40.0,
            "R_tile_to_coolant_K_W": 0.8,
            "R_hbm_to_coolant_K_W": 1.2,
            "R_vrm_to_coolant_K_W": 1.5
        }
        coolant_config = {"T_in_C": 40.0}
        
        issues = validate_chip_numerical_stability(
            gpu_config, coolant_config, dt=0.5)
        
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0, f"Unexpected stability errors: {errors}"
    
    def test_unstable_timestep_detected(self):
        """Unstable timestep should be detected."""
        gpu_config = {
            "C_tile_J_K": 1.0,  # Very small capacitance
            "C_hbm_J_K": 10.0,
            "C_vrm_J_K": 5.0,
            "R_tile_to_coolant_K_W": 0.1,  # Low resistance = fast dynamics
            "R_hbm_to_coolant_K_W": 0.2,
            "R_vrm_to_coolant_K_W": 0.3
        }
        coolant_config = {"T_in_C": 40.0}
        
        issues = validate_chip_numerical_stability(
            gpu_config, coolant_config, dt=5.0)  # Large timestep
        
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) > 0, "Should detect unstable timestep"


# ══════════════════════════════════════════════════════════════════════════════
# Test CDU Validation
# ══════════════════════════════════════════════════════════════════════════════

class TestCDUValidation:
    """Test CDU configuration validation."""
    
    def test_stable_cdu_config(self):
        """Stable CDU configuration should pass."""
        cdu_config = {
            "UA_W_K": 5000,
            "hot_loop_V_m3": 0.05,
            "cold_loop_V_m3": 0.10
        }
        
        issues = validate_cdu_sizing(cdu_config, expected_heat_load_W=20000, dt=5.0)
        
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0, f"Unexpected CDU errors: {errors}"
    
    def test_unstable_cdu_detected(self):
        """Unstable CDU configuration should be detected."""
        cdu_config = {
            "UA_W_K": 50000,  # Very high UA
            "hot_loop_V_m3": 0.001,  # Very small volume
            "cold_loop_V_m3": 0.002
        }
        
        issues = validate_cdu_sizing(cdu_config, expected_heat_load_W=20000, dt=5.0)
        
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) > 0, "Should detect unstable CDU"


# ══════════════════════════════════════════════════════════════════════════════
# Test Cooling Tower Validation
# ══════════════════════════════════════════════════════════════════════════════

class TestCoolingTowerValidation:
    """Test cooling tower configuration validation."""
    
    def test_stable_tower_config(self):
        """Stable cooling tower configuration should pass."""
        tower_config = {
            "T_ambient_C": 25.0,
            "T_wetbulb_C": 18.0,
            "fan_UA_W_K": 3000,
            "basin_C_J_K": 2000000,
            "evap_fraction": 0.75,
            "latent_heat_J_kg": 2260000
        }
        
        issues = validate_cooling_tower_sizing(
            tower_config, expected_heat_load_W=20000, dt=5.0)
        
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0, f"Unexpected tower errors: {errors}"
    
    def test_unstable_tower_detected(self):
        """Unstable cooling tower configuration should be detected."""
        tower_config = {
            "T_ambient_C": 25.0,
            "T_wetbulb_C": 18.0,
            "fan_UA_W_K": 30000,  # High UA
            "basin_C_J_K": 10000,  # Very small basin
            "evap_fraction": 0.75,
            "latent_heat_J_kg": 2260000
        }
        
        issues = validate_cooling_tower_sizing(
            tower_config, expected_heat_load_W=20000, dt=5.0)
        
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) > 0, "Should detect unstable cooling tower"


# ══════════════════════════════════════════════════════════════════════════════
# Test System-Level Validation
# ══════════════════════════════════════════════════════════════════════════════

class TestSystemValidation:
    """Test full system validation."""
    
    def test_valid_small_system(self):
        """Small valid system should pass validation."""
        result = validate_system_config(
            gpu_preset="H100_SXM",
            n_gpus=8,
            cdu_preset="small_datacenter",
            tower_preset="small_tower",
            dt=5.0,
            target_temp_C=75.0
        )
        assert result.is_valid
    
    def test_invalid_preset_fails(self):
        """Invalid preset should fail validation."""
        result = validate_system_config(
            gpu_preset="NONEXISTENT_GPU",
            n_gpus=8,
            cdu_preset="small_datacenter",
            tower_preset="small_tower",
            dt=5.0
        )
        assert not result.is_valid
        assert any(i.severity == Severity.CRITICAL for i in result.issues)
    
    def test_validation_result_properties(self):
        """ValidationResult should have proper properties."""
        result = validate_system_config(
            gpu_preset="H100_SXM",
            n_gpus=8,
            cdu_preset="small_datacenter",
            tower_preset="small_tower",
            dt=5.0
        )
        
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert isinstance(str(result), str)


# ══════════════════════════════════════════════════════════════════════════════
# Test Edge Cases
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_flow_rate(self):
        """Zero flow rate should be handled."""
        gpu_config = load_component_config("gpu_presets", "H100_SXM")
        coolant_config = {
            "T_in_C": 40.0,
            "m_dot_min_kg_s": 0.0,  # Zero min flow
            "m_dot_max_kg_s": 0.5,
            "c_p_J_kgK": 800.0
        }
        
        # Should not crash, but should flag issues
        issues = validate_chip_thermal_feasibility(
            gpu_config, coolant_config, target_temp_C=70.0)
        assert isinstance(issues, list)
    
    def test_very_small_timestep(self):
        """Very small timestep should be valid."""
        result = validate_chip_env_config(
            gpu_preset="H100_SXM",
            dt=0.001,  # 1 ms
            target_temp_C=70.0
        )
        # Should be valid (maybe info about being conservative)
        errors = [i for i in result.issues if i.severity == Severity.ERROR]
        assert len(errors) == 0
    
    def test_very_large_heat_load(self):
        """Very large heat load should produce warnings."""
        result = validate_system_config(
            gpu_preset="H100_SXM",
            n_gpus=1000,  # Very large system
            cdu_preset="small_datacenter",  # But small CDU
            tower_preset="small_tower",
            dt=5.0
        )
        # Should have sizing warnings at least
        warnings = [i for i in result.issues if i.severity == Severity.WARNING]
        assert len(warnings) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Integration Tests with Actual Environments
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegrationWithEnvironments:
    """Test that valid configurations actually work in simulation."""
    
    def test_valid_chip_config_runs(self):
        """Valid chip configuration should run without numerical issues."""
        import compopt
        
        # Validate first
        result = validate_chip_env_config(
            gpu_preset="H100_SXM",
            m_dot_min=0.05,
            m_dot_max=0.5,
            c_p=800.0,
            dt=0.5,
            target_temp_C=70.0
        )
        assert result.is_valid
        
        # Then run simulation
        env = compopt.make("ChipThermal-v0", dt=0.5)
        obs, _ = env.reset(seed=42)
        
        temps = []
        for _ in range(100):
            action = env.action_space.sample()
            obs, _, terminated, truncated, info = env.step(action)
            temps.append(info["T_hotspot_C"])
            
            if terminated or truncated:
                break
        
        env.close()
        
        # Check temperatures are reasonable (not blown up)
        assert all(20 < t < 200 for t in temps), \
            f"Temperature out of reasonable range: {min(temps):.1f} - {max(temps):.1f}"
        assert not any(np.isnan(temps) or np.isinf(temps) for temps in temps)
    
    def test_valid_datacenter_config_runs(self):
        """Valid datacenter configuration should run without issues."""
        import compopt
        
        # Validate first
        result = validate_system_config(
            gpu_preset="H100_SXM",
            n_gpus=16,
            cdu_preset="small_datacenter",
            tower_preset="small_tower",
            dt=5.0
        )
        assert result.is_valid
        
        # Then run simulation
        env = compopt.make("DataCenter-v0", dt=5.0, n_racks=2, 
                          servers_per_rack=4, gpus_per_server=2)
        obs, _ = env.reset(seed=42)
        
        pues = []
        wues = []
        for _ in range(50):
            action = env.action_space.sample()
            obs, _, terminated, truncated, info = env.step(action)
            pues.append(info.get("PUE", 0))
            wues.append(info.get("WUE_L_per_kWh", 0))
            
            if terminated or truncated:
                break
        
        env.close()
        
        # Check metrics are reasonable
        assert all(1.0 < p < 10.0 for p in pues), \
            f"PUE out of range: {min(pues):.2f} - {max(pues):.2f}"
        assert all(0 < w < 50 for w in wues), \
            f"WUE out of range: {min(wues):.2f} - {max(wues):.2f}"


# ══════════════════════════════════════════════════════════════════════════════
# Test Helpful Error Messages
# ══════════════════════════════════════════════════════════════════════════════

class TestHelpfulMessages:
    """Test that validation provides helpful feedback."""
    
    def test_issues_have_suggestions(self):
        """Warning/error issues should have suggestions."""
        # Create a configuration with known issues
        gpu_config = {
            "tdp_W": 1000,
            "nx_tiles": 4,
            "ny_tiles": 4,
            "R_tile_to_coolant_K_W": 3.0,  # Too high
            "T_max_die_C": 83.0
        }
        coolant_config = {
            "T_in_C": 40.0,
            "m_dot_min_kg_s": 0.1,
            "m_dot_max_kg_s": 0.2,  # Low capacity
            "c_p_J_kgK": 800.0
        }
        
        issues = validate_chip_thermal_feasibility(
            gpu_config, coolant_config, target_temp_C=70.0)
        
        # Issues with warnings/errors should have suggestions
        serious_issues = [i for i in issues 
                        if i.severity in (Severity.WARNING, Severity.ERROR)]
        for issue in serious_issues:
            # Either has suggestion or is informational
            assert issue.suggestion is not None or "info" in issue.category.lower(), \
                f"Issue lacks suggestion: {issue}"
    
    def test_validation_result_string_format(self):
        """Validation result should have readable string format."""
        result = validate_system_config(
            gpu_preset="H100_SXM",
            n_gpus=8,
            cdu_preset="small_datacenter",
            tower_preset="small_tower",
            dt=5.0
        )
        
        result_str = str(result)
        assert "validation" in result_str.lower()
        assert "VALID" in result_str or "INVALID" in result_str
