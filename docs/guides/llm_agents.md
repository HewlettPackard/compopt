# LLM Agents

CompOpt supports **agentic AI** controllers that use large language models
(LLMs) with retrieval-augmented generation (RAG) to make cooling and
scheduling decisions.

## Architecture

```
┌───────────┐     ┌──────────┐     ┌─────────────┐
│ CompOpt Env│────▸│ Obs → JSON│────▸│ LangChain   │
│           │     │ Formatter │     │ RAG Chain   │
│           │◂────│ JSON → Act│◂────│ + vLLM      │
└───────────┘     └──────────┘     └─────────────┘
                                        │
                                   ┌────┴────┐
                                   │ FAISS   │
                                   │ Vector  │
                                   │ Store   │
                                   └─────────┘
                                        ▲
                                   ┌────┴────┐
                                   │ RAG Doc │
                                   │ (txt)   │
                                   └─────────┘
```

## Prerequisites

```bash
pip install -e ".[llm]"

# Start a vLLM server (in a separate terminal):
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --port 8000
```

## LLM Cooling Agent

Controls rack/datacenter cooling via natural language reasoning:

```python
from compopt.agents.llm_agent import LLMCoolingAgent
import compopt

env = compopt.make("RackCooling-v0")
agent = LLMCoolingAgent(
    docs_path="docs/compopt_reference.txt",
    vllm_base_url="http://localhost:8000/v1",
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
)

obs, info = env.reset()
for step in range(100):
    action, meta = agent.predict(obs)
    obs, reward, term, trunc, info = env.step(action)
    print(f"Step {step}: action={action[0]:.2f}, T={info['T_hotspot_C']:.1f}°C")
```

### How It Works

1. Observation is converted to a JSON-serialisable dict
2. System prompt + observation + RAG context are sent to vLLM
3. LLM responds with a JSON action: `{"action_norm": 0.6, "reason": "..."}`
4. Response is parsed and clipped to [0, 1]

## LLM Joint Agent

For `JointDCFlat-v0` — controls both cooling and scheduling:

```python
from compopt.agents.llm_agent import LLMJointAgent

env = compopt.make("JointDCFlat-v0")
agent = LLMJointAgent(
    docs_path="docs/compopt_reference.txt",
    vllm_base_url="http://localhost:8000/v1",
)

obs, info = env.reset()
action, meta = agent.predict(obs)
# action = [rack_flow, cdu_pump, tower_fan, sched_index/10]
```

## RAG Grounding Document

The RAG chain uses `docs/compopt_reference.txt` as its knowledge base.
This document contains:

- System description and thermal limits
- Observation vector definitions
- Action space semantics
- Control heuristics and safety rules
- Physics equations and parameters

You can customise or extend this document for domain-specific knowledge.

## Custom System Prompts

Override the default prompts:

```python
my_prompt = """
You are a cooling controller. Keep T < 75°C at all costs.
Respond with: {"action_norm": <float>}
"""

agent = LLMCoolingAgent(system_prompt=my_prompt)
```

## Building Custom RAG Chains

```python
from compopt.agents.llm_agent import build_rag_chain

chain = build_rag_chain(
    docs_path="my_custom_docs.txt",
    vllm_base_url="http://localhost:8000/v1",
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    temperature=0.05,
)

result = chain.run("What is the thermal limit for H100 GPUs?")
```

## Comparing LLM vs RL Agents

A key research question for the benchmark:

```python
from compopt.utils import evaluate_agent, print_metrics
from compopt.agents import PIDCoolingAgent
from compopt.agents.llm_agent import LLMCoolingAgent

env = compopt.make("RackCooling-v0")

pid_metrics = evaluate_agent(env, PIDCoolingAgent(), n_episodes=5)
llm_metrics = evaluate_agent(env, LLMCoolingAgent(), n_episodes=5)

print_metrics(pid_metrics, "PID Agent")
print_metrics(llm_metrics, "LLM Agent")
```
