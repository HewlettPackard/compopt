"""
CompOpt Test Suite — Physics Models
====================================
Tests for chip, server, rack, and datacenter thermal models.
"""

import numpy as np
import pytest

from compopt.physics.fluids import CoolantLoop, CDU, CoolingTower, BatchCoolantLoop
from compopt.physics.chip import GPUChipModel, PowerProfile, make_gpu, GPU_PRESETS
from compopt.physics.workloads import (
    make_sinusoidal_profile, make_step_profile, make_burst_profile,
    make_stochastic_profile, make_mixed_profile,
)
from compopt.physics.server import (
    ServerNode, RackModel, DataCenterModel,
    build_default_rack, build_default_datacenter,
)


# ──────────────────────────────────────────────────────────────────────────────
# CoolantLoop
# ──────────────────────────────────────────────────────────────────────────────

class TestCoolantLoop:
    def test_init(self):
        loop = CoolantLoop(T_in_C=40.0, m_dot_kg_s=0.5)
        assert loop.T_coolant_C == 40.0
        assert loop.m_dot_kg_s == 0.5

    def test_step_heats_coolant(self):
        loop = CoolantLoop(T_in_C=40.0, m_dot_kg_s=0.5)
        for _ in range(100):
            loop.step(1.0, Q_in_W=1000.0)
        assert loop.T_coolant_C > 40.0

    def test_reset(self):
        loop = CoolantLoop(T_in_C=40.0, m_dot_kg_s=0.5)
        loop.step(1.0, Q_in_W=1000.0)
        assert loop.T_coolant_C > 40.0
        loop.reset()
        assert loop.T_coolant_C == 40.0

    def test_zero_flow_doesnt_crash(self):
        loop = CoolantLoop(T_in_C=40.0, m_dot_kg_s=0.0)
        loop.step(1.0, Q_in_W=500.0)  # should not divide by zero


class TestBatchCoolantLoop:
    def test_batch_stepping(self):
        batch = BatchCoolantLoop(batch_size=4, T_in_C=40.0, m_dot_kg_s=0.5)
        Q = np.array([500, 1000, 1500, 2000], dtype=np.float64)
        for _ in range(50):
            batch.step(1.0, Q_in_W=Q)
        # Higher heat load → higher temperature
        T = batch.T_coolant
        assert T[3] > T[0]


class TestCDU:
    def test_init(self):
        cdu = CDU(T_facility_C=25.0)
        assert cdu.T_facility_C == 25.0

    def test_heat_exchange(self):
        cdu = CDU(T_facility_C=25.0)
        for _ in range(100):
            cdu.step(1.0, Q_rack_total_W=50000)
        assert cdu.hot_loop.T_coolant_C > 25.0


class TestCoolingTower:
    def test_init(self):
        tower = CoolingTower(T_ambient_C=25.0, T_wetbulb_C=18.0)
        assert tower.T_ambient_C == 25.0

    def test_stepping_tracks_water(self):
        tower = CoolingTower(T_ambient_C=25.0, T_wetbulb_C=18.0)
        for _ in range(100):
            tower.step(1.0, Q_facility_W=30000)
        assert tower.water_used_L >= 0.0

    def test_reset_clears_water(self):
        tower = CoolingTower(T_ambient_C=25.0, T_wetbulb_C=18.0)
        tower.step(1.0, Q_facility_W=30000)
        tower.reset()
        assert tower.water_used_L == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# GPU Chip Model
# ──────────────────────────────────────────────────────────────────────────────

class TestGPUChipModel:
    def test_make_gpu_h100(self):
        gpu = make_gpu("H100_SXM")
        assert gpu.nx_tiles == 4
        assert gpu.ny_tiles == 4
        assert len(gpu.node_names) == 4 * 4 + 3  # tiles + HBM + VRM + PCB

    def test_make_gpu_b200(self):
        gpu = make_gpu("B200")
        assert gpu.nx_tiles == 6
        assert gpu.ny_tiles == 6
        assert len(gpu.node_names) == 6 * 6 + 3

    def test_step_changes_temperature(self):
        gpu = make_gpu("H100_SXM")
        T_initial = gpu.T_C.copy()
        for _ in range(100):
            gpu.step(1.0)
        assert not np.allclose(gpu.T_C, T_initial)

    def test_observation_shape(self):
        gpu = make_gpu("H100_SXM")
        obs = gpu.get_observation()
        assert obs.shape == (7,)

    def test_sensors_dict_keys(self):
        gpu = make_gpu("H100_SXM")
        gpu.step(1.0)
        d = gpu.get_sensors_dict()
        assert "T_gpu_hotspot_C" in d
        assert "T_hbm_C" in d
        assert "time_s" in d

    def test_reset_restores_state(self):
        gpu = make_gpu("H100_SXM")
        for _ in range(50):
            gpu.step(1.0)
        gpu.reset()
        assert gpu.time_s == 0.0
        assert gpu.T_C[0] == gpu.T_init_die_C

    def test_is_throttled_false_at_init(self):
        gpu = make_gpu("H100_SXM")
        assert not gpu.is_throttled()

    def test_make_gpu_with_overrides(self):
        gpu = make_gpu("H100_SXM", T_init_die_C=50.0)
        assert gpu.T_C[0] == 50.0

    def test_rk4_stability(self):
        """RK4 should not blow up even with large time steps."""
        gpu = make_gpu("H100_SXM")
        for _ in range(20):
            gpu.step(5.0)  # larger time step
        assert np.all(np.isfinite(gpu.T_C))


# ──────────────────────────────────────────────────────────────────────────────
# Workloads
# ──────────────────────────────────────────────────────────────────────────────

class TestWorkloads:
    def test_sinusoidal_profile(self):
        profile = make_sinusoidal_profile(4, 4, base_gpu_W=100, peak_gpu_W=700)
        assert isinstance(profile, PowerProfile)
        P = profile.power_vector(0.0, [f"tile_{r}_{c}" for r in range(4) for c in range(4)])
        assert len(P) == 16
        assert np.all(P >= 0)

    def test_step_profile(self):
        profile = make_step_profile(4, 4)
        P0 = profile.power_vector(0.0, ["tile_0_0"])
        P300 = profile.power_vector(300.0, ["tile_0_0"])
        # Step profiles should change between phases
        assert isinstance(P0[0], (float, np.floating))

    def test_burst_profile(self):
        profile = make_burst_profile(4, 4)
        P = profile.power_vector(100.0, ["tile_0_0"])
        assert len(P) == 1

    def test_stochastic_profile(self):
        profile = make_stochastic_profile(4, 4, seed=42)
        P1 = profile.power_vector(10.0, ["tile_0_0"])
        P2 = profile.power_vector(200.0, ["tile_0_0"])
        assert isinstance(P1[0], (float, np.floating))

    def test_mixed_profile(self):
        sin_p = make_sinusoidal_profile(4, 4)
        step_p = make_step_profile(4, 4)
        mixed = make_mixed_profile(4, 4, profiles=[sin_p, step_p], weights=[0.6, 0.4])
        P = mixed.power_vector(50.0, ["tile_0_0", "HBM"])
        assert len(P) == 2


# ──────────────────────────────────────────────────────────────────────────────
# Server / Rack / DataCenter
# ──────────────────────────────────────────────────────────────────────────────

class TestServerNode:
    def test_creation(self):
        gpu = make_gpu("H100_SXM")
        server = ServerNode(gpus=[gpu], node_id=0)
        assert server.P_total_W > 0

    def test_step(self):
        gpu = make_gpu("H100_SXM")
        server = ServerNode(gpus=[gpu])
        server.step(1.0)
        assert server.T_gpu_hotspot_C > 0

    def test_telemetry(self):
        gpu = make_gpu("H100_SXM")
        server = ServerNode(gpus=[gpu])
        tele = server.get_telemetry()
        assert "P_total_W" in tele
        assert "T_gpu_hotspot_C" in tele


class TestRackModel:
    def test_build_default(self):
        rack = build_default_rack(n_servers=2, gpus_per_server=1)
        assert len(rack.servers) == 2

    def test_step(self):
        rack = build_default_rack(n_servers=2, gpus_per_server=1)
        rack.step(1.0)
        assert rack.time_s == 1.0

    def test_observation_shape(self):
        rack = build_default_rack(n_servers=2, gpus_per_server=1)
        obs = rack.get_rack_observation()
        assert obs.shape == (10,)

    def test_power_positive(self):
        rack = build_default_rack(n_servers=2, gpus_per_server=1)
        assert rack.P_total_W > 0


class TestDataCenterModel:
    def test_build_default(self):
        dc = build_default_datacenter(n_racks=2, servers_per_rack=2,
                                       gpus_per_server=1)
        assert len(dc.racks) == 2

    def test_step(self):
        dc = build_default_datacenter(n_racks=1, servers_per_rack=2,
                                       gpus_per_server=1)
        dc.step(5.0)
        assert dc.time_s == 5.0

    def test_observation_shape(self):
        dc = build_default_datacenter(n_racks=1, servers_per_rack=2,
                                       gpus_per_server=1)
        obs = dc.get_observation()
        assert obs.shape == (16,)

    def test_pue_reasonable(self):
        dc = build_default_datacenter(n_racks=1, servers_per_rack=2,
                                       gpus_per_server=1)
        dc.step(1.0)
        assert dc.PUE >= 1.0

    def test_reset(self):
        dc = build_default_datacenter(n_racks=1, servers_per_rack=2,
                                       gpus_per_server=1)
        for _ in range(10):
            dc.step(1.0)
        dc.reset()
        assert dc.time_s == 0.0
        assert dc._energy_J == 0.0
