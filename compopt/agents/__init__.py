"""
compopt.agents
==============

Built-in baseline and agentic AI control agents for benchmarking.

Sub-modules
-----------
baselines
    Rule-based, PID, constant, and random agents.
llm_agent
    LLM-based (vLLM + LangChain RAG) cooling and joint controllers.
"""

from compopt.agents.baselines import (
    RandomAgent, ConstantAgent,
    RuleBasedCoolingAgent, PIDCoolingAgent,
    DataCenterRuleAgent, FCFSSchedulingAgent,
)

__all__ = [
    "RandomAgent", "ConstantAgent",
    "RuleBasedCoolingAgent", "PIDCoolingAgent",
    "DataCenterRuleAgent", "FCFSSchedulingAgent",
]
