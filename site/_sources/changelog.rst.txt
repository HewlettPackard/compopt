Changelog
=========

v0.1.0 — Initial Release
------------------------

*2025-07-17*

Features
^^^^^^^^

- **Physics engine** — RC-network GPU thermal model with RK4 auto sub-stepping,
  semi-implicit coolant loop, CDU (UA-method), and evaporative cooling tower.
- **GPU profiles** — NVIDIA H100 SXM (700 W, 4×4 grid) and B200 (1000 W, 6×6 grid).
- **Workload generators** — Constant, sinusoidal, step, burst, and random-walk profiles.
- **6 Gymnasium environments**:
  - ``ChipThermal-v0`` — Single-chip flow-rate control.
  - ``RackCooling-v0`` — Multi-GPU rack cooling.
  - ``Scheduling-v0`` — Job scheduling with thermal awareness.
  - ``DataCenter-v0`` — Full data-center cooling (CDU + tower).
  - ``JointDC-v0`` — Joint scheduling + cooling (``Dict`` action space).
  - ``JointDCFlat-v0`` — Flattened variant of ``JointDC-v0``.
- **Reward components** — Temperature penalty, energy cost, PUE bonus, water cost,
  throttle penalty, SLA penalty, and composite reward with presets.
- **Baseline agents** — Fixed, random, proportional, PID, threshold, and LLM-based.
- **Utilities** — Batch simulator, episode metrics, and Matplotlib plotting helpers.
- **Configuration** — JSON-based presets (``small_rack``, ``standard_dc``, ``large_dc``,
  ``energy_focused``, ``water_focused``) with ``build_from_config`` factory.
- **Numerical stability** — Semi-implicit coolant integration and automatic RK4
  sub-stepping with CFL-like stability bound.
