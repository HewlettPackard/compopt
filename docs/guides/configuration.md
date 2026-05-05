# Configuration Presets

CompOpt ships with JSON configuration presets that define complete
data-center setups — number of racks, servers, GPUs, cooling parameters,
power costs, and scheduler settings.

## Available Presets

```python
from compopt.configs import list_presets, load_config

print(list_presets())
# ['large_cloud_dc', 'medium_hpc_center', 'small_ai_cluster']
```

| Preset              | Racks | Nodes | GPUs/Node | GPU Type  | Description             |
|---------------------|------:|------:|----------:|-----------|-------------------------|
| `small_ai_cluster`  |     2 |     8 |         2 | H100_SXM  | Small research lab      |
| `medium_hpc_center` |     8 |    64 |         4 | H100_SXM  | Medium HPC facility     |
| `large_cloud_dc`    |    32 |   256 |         8 | B200      | Large cloud data center |

## Loading a Preset

```python
from compopt.configs import load_config

cfg = load_config("small_ai_cluster")
print(cfg.keys())
# dict_keys(['system', 'power', 'scheduler', 'cooling'])
```

## Building from Config

`build_from_config` creates a fully instantiated `DataCenterModel` +
`Scheduler` + job list from a named preset:

```python
from compopt.configs import build_from_config

result = build_from_config("small_ai_cluster", n_jobs=100, seed=42)

dc = result["datacenter"]       # DataCenterModel
scheduler = result["scheduler"]  # Scheduler
jobs = result["jobs"]            # List[Job]
config = result["config"]       # raw JSON dict

# Use in a simulation loop
scheduler.submit_batch(jobs)
for t in range(3600):
    scheduler.step(dt=1.0)
    dc.step(dt=1.0)
```

## JSON Format

Each preset JSON file has four sections:

```json
{
  "system": {
    "NUM_CDUS": 1,
    "RACKS_PER_CDU": 2,
    "NODES_PER_RACK": 4,
    "GPUS_PER_NODE": 2,
    "GPU_PRESET": "H100_SXM"
  },
  "power": {
    "POWER_CPU_IDLE": 90,
    "POWER_CPU_MAX": 280,
    "POWER_MEM": 74,
    "POWER_NIC": 20,
    "POWER_NVME": 30,
    "POWER_COST": 0.094
  },
  "scheduler": {
    "SEED": 42,
    "MAX_NODES_PER_JOB": 8,
    "JOB_ARRIVAL_TIME": 900
  },
  "cooling": {
    "CDU_UA_W_K": 50000,
    "T_FACILITY_C": 30.0,
    "TOWER_FAN_POWER_W": 15000,
    "WATER_COST_DOLLAR_PER_L": 0.004
  }
}
```

## Custom Presets

Add your own preset by creating a JSON file in `compopt/configs/`:

```python
import json
from pathlib import Path

my_config = {
    "system": {"NUM_CDUS": 2, "RACKS_PER_CDU": 4, ...},
    "power": {...},
    "scheduler": {...},
    "cooling": {...},
}

config_dir = Path("compopt/configs")
with open(config_dir / "my_custom_dc.json", "w") as f:
    json.dump(my_config, f, indent=2)

# Now available:
from compopt.configs import build_from_config
result = build_from_config("my_custom_dc")
```
