"""
compopt.utils.plotting
======================
Publication-quality plotting utilities for CompOpt experiments.

Generates figures suitable for NeurIPS papers: thermal profiles,
cooling actuator traces, reward curves, PUE/WUE timelines, and
scheduling Gantt charts.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional


def plot_thermal_episode(metrics,
                         title: str = "Thermal Episode",
                         save_path: Optional[str] = None,
                         target_C: float = 80.0):
    """
    Plot thermal + flow + reward over an episode.

    Parameters
    ----------
    metrics : EpisodeMetrics instance (from compopt.utils.metrics)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    t = np.arange(len(metrics.T_hotspot_history))

    # Temperature
    ax = axes[0]
    ax.plot(t, metrics.T_hotspot_history, label="GPU Hotspot", color="tab:red")
    ax.axhline(target_C, color="gray", linestyle="--", label=f"Target {target_C}°C")
    ax.axhline(target_C + 3, color="red", linestyle=":", alpha=0.5,
               label=f"Limit {target_C+3}°C")
    ax.set_ylabel("Temperature [°C]")
    ax.legend(fontsize=9)
    ax.set_title(title)

    # Flow
    ax = axes[1]
    ax.plot(t, metrics.flow_history, label="Coolant Flow", color="tab:blue")
    ax.set_ylabel("Flow [kg/s]")
    ax.legend(fontsize=9)

    # Reward
    ax = axes[2]
    ax.plot(t, np.cumsum(metrics.reward_history),
            label="Cumulative Reward", color="tab:green")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_datacenter_episode(metrics,
                             title: str = "Data Center Episode",
                             save_path: Optional[str] = None):
    """
    Plot datacenter-level metrics: PUE, WUE, cost, power, temperature.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    t = np.arange(len(metrics.T_hotspot_history))

    # Temperature
    ax = axes[0]
    ax.plot(t, metrics.T_hotspot_history, color="tab:red", label="T_hotspot")
    ax.axhline(80, color="gray", linestyle="--")
    ax.set_ylabel("Temperature [°C]")
    ax.legend()
    ax.set_title(title)

    # PUE & WUE
    ax = axes[1]
    ax.plot(t, metrics.pue_history, color="tab:blue", label="PUE")
    ax2 = ax.twinx()
    ax2.plot(t, metrics.wue_history, color="tab:cyan", label="WUE [L/kWh]")
    ax.set_ylabel("PUE")
    ax2.set_ylabel("WUE [L/kWh]")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2)

    # Power
    ax = axes[2]
    ax.plot(t, metrics.power_history, color="tab:orange", label="Total Power [kW]")
    ax.set_ylabel("Power [kW]")
    ax.legend()

    # Cost & Water
    ax = axes[3]
    ax.plot(t, np.cumsum(metrics.cost_history), color="tab:green",
            label="Cumulative Cost [$]")
    ax2 = ax.twinx()
    water_cumulative = np.cumsum([0.0] * len(t))  # placeholder
    ax2.plot(t, water_cumulative, color="tab:purple",
             label="Cumulative Water [L]", alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cost [$]")
    ax2.set_ylabel("Water [L]")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_comparison(results: Dict[str, Dict[str, float]],
                    metrics_to_plot: Optional[List[str]] = None,
                    title: str = "Agent Comparison",
                    save_path: Optional[str] = None):
    """
    Bar chart comparing multiple agents across key metrics.

    Parameters
    ----------
    results : {agent_name: metrics_dict}
    """
    import matplotlib.pyplot as plt

    if metrics_to_plot is None:
        metrics_to_plot = [
            "T_hotspot_mean", "reward_total", "PUE_mean",
            "total_water_L", "total_cost_dollar",
        ]

    n_metrics = len(metrics_to_plot)
    n_agents  = len(results)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    agents = list(results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, n_agents))

    for ax, metric in zip(axes, metrics_to_plot):
        vals = [results[a].get(metric, 0.0) for a in agents]
        stds = [results[a].get(f"{metric}_std", 0.0) for a in agents]
        bars = ax.bar(agents, vals, yerr=stds, color=colors[:n_agents],
                       capsize=3, edgecolor="gray")
        ax.set_title(metric.replace("_", " ").title(), fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=8)

    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
