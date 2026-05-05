#!/usr/bin/env python3
"""
CompOpt Example 08 — LLM / Agentic AI Cooling Control
======================================================
Demonstrates the LLM agent interface for natural-language-based cooling
control using vLLM + LangChain with RAG grounding.

**Requires**: pip install 'compopt[llm]'
              A running vLLM server or OpenAI API key

Run:
    python examples/08_llm_agent.py --api-base http://localhost:8000/v1 --model mistralai/Mixtral-8x7B-v0.1
"""

import argparse
import numpy as np

import compopt
from compopt.utils.metrics import EpisodeMetrics, print_metrics

# ── Parse arguments ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="CompOpt LLM Agent Demo")
parser.add_argument("--api-base", type=str,
                    default="http://localhost:8000/v1",
                    help="OpenAI-compatible API base URL (e.g. vLLM server)")
parser.add_argument("--model", type=str,
                    default="mistralai/Mixtral-8x7B-v0.1",
                    help="Model name for the LLM")
parser.add_argument("--env", type=str, default="DataCenter-v0",
                    choices=["DataCenter-v0", "RackCooling-v0", "ChipThermal-v0"],
                    help="Environment to control")
parser.add_argument("--steps", type=int, default=100,
                    help="Number of control steps")
parser.add_argument("--api-key", type=str, default="EMPTY",
                    help="API key (use EMPTY for local vLLM)")
args = parser.parse_args()


def run_with_fallback():
    """Try LLM agent, fall back to rule-based if LLM deps missing."""
    try:
        from compopt.agents.llm_agent import LLMCoolingAgent
        HAS_LLM = True
    except ImportError:
        HAS_LLM = False
        print("LLM dependencies not installed. Install with:")
        print("  pip install 'compopt[llm]'")
        print("\nFalling back to DataCenterRuleAgent for demonstration.\n")

    env = compopt.make(args.env, dt=5.0, episode_length_s=args.steps * 5.0)

    if HAS_LLM:
        import os
        os.environ["OPENAI_API_BASE"] = args.api_base
        os.environ["OPENAI_API_KEY"] = args.api_key

        agent = LLMCoolingAgent(
            docs_path="docs/compopt_reference.txt",
            vllm_base_url=args.api_base,
            model_name=args.model,
        )
        agent_name = f"LLM ({args.model})"
    else:
        from compopt.agents.baselines import DataCenterRuleAgent
        agent = DataCenterRuleAgent(target_C=80.0)
        agent_name = "Rule-Based (fallback)"

    metrics = EpisodeMetrics()
    obs, _ = env.reset()
    if hasattr(agent, 'reset'):
        agent.reset()

    print(f"Running {agent_name} on {args.env}")
    print("-" * 60)

    for step in range(args.steps):
        action, info_agent = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        metrics.record(info, reward, time_s=step * 5.0)

        if step % 20 == 0:
            T = info.get("T_hotspot_C", info.get("T_gpu_hotspot_C", 0))
            print(f"  Step {step:4d}  T_hot={T:.1f}°C  reward={reward:.3f}")

            # Show LLM reasoning if available
            if HAS_LLM and "llm_reasoning" in info_agent:
                reasoning = info_agent["llm_reasoning"][:80]
                print(f"    LLM: {reasoning}...")

        if terminated or truncated:
            break

    print_metrics(metrics.summary(), title=f"{agent_name} on {args.env}")
    env.close()


if __name__ == "__main__":
    run_with_fallback()
