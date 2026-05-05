"""
compopt.agents.llm_agent
========================
LLM-based (agentic AI) controller using vLLM + LangChain RAG.

The ``LLMCoolingAgent`` and ``LLMJointAgent`` query a local vLLM server
at each control step, providing the current observation + RAG context
from reference documentation, and parse the JSON response into actions.

Prerequisites::

    pip install langchain langchain-community sentence-transformers faiss-cpu openai

    # Start vLLM server:
    python -m vllm.entrypoints.openai.api_server \\
        --model meta-llama/Meta-Llama-3-8B-Instruct --port 8000
"""

from __future__ import annotations

import json
import numpy as np
from typing import Any, Dict, List, Optional
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# System prompts
# ──────────────────────────────────────────────────────────────────────────────

COOLING_SYSTEM_PROMPT = """
You are an AI controller for a liquid-cooled GPU data center simulator (CompOpt).

Your goal: keep GPU hotspot temperature below 80°C while minimising coolant flow
(to save pumping energy), water consumption, and operating cost.

Action: a single float "action_norm" in [0, 1].
  Actual rack coolant flow = 0.5 + action_norm * 3.5  (kg/s)

Observation fields:
  T_gpu_hotspot_C : GPU die hotspot temperature [°C] — primary safety signal
  T_hbm_C         : HBM junction temperature [°C]
  T_vrm_C         : VRM temperature [°C]
  P_total_W       : total power draw [W]
  flow_kg_s       : current coolant flow [kg/s]

Control heuristics:
  - If hotspot > 82°C: increase flow (action_norm → 1.0)
  - If hotspot 78–82°C: maintain flow
  - If hotspot < 75°C: reduce flow (action_norm → 0.2)
  - At idle (<200W): action_norm ≈ 0.1
  - At full load (>700W): action_norm ≥ 0.6

Respond ONLY with a single-line JSON:
{"action_norm": <float 0–1>, "reason": "<brief explanation>"}
"""

JOINT_SYSTEM_PROMPT = """
You are an AI controller for joint scheduling + cooling in a GPU data center.

You must simultaneously:
1. Control cooling: set rack flow, CDU pump, and tower fan levels.
2. Schedule jobs: prioritise which queued job to run next.

Cooling actions (3 floats, each [0, 1]):
  rack_flow_norm  : rack coolant flow  (0.5 + v * 3.5 kg/s)
  cdu_pump_norm   : CDU pump flow      (1.0 + v * 7.0 kg/s)
  tower_fan_norm  : cooling tower fan  (5000 + v * 25000 W)

Scheduling action:
  queue_index     : integer 0–10. 0 = no-op, 1–10 = move that queue slot to front.

Respond ONLY with JSON:
{"rack_flow_norm": <float>, "cdu_pump_norm": <float>, "tower_fan_norm": <float>,
 "queue_index": <int>, "reason": "<brief>"}
"""


# ──────────────────────────────────────────────────────────────────────────────
# RAG chain builder
# ──────────────────────────────────────────────────────────────────────────────

def build_rag_chain(docs_path: str = "docs/h100_thermal_notes.txt",
                    vllm_base_url: str = "http://localhost:8000/v1",
                    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                    temperature: float = 0.1):
    """
    Build a LangChain RAG chain backed by a local vLLM server.

    Parameters
    ----------
    docs_path     : path to the RAG grounding document
    vllm_base_url : vLLM OpenAI-compatible API endpoint
    model_name    : model served by vLLM
    temperature   : LLM sampling temperature

    Returns
    -------
    RetrievalQA chain
    """
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.chat_models import ChatOpenAI
    from langchain.chains import RetrievalQA

    loader  = TextLoader(docs_path, encoding="utf-8")
    docs    = loader.load()
    chunks  = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100).split_documents(docs)
    emb     = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    db      = FAISS.from_documents(chunks, emb)
    retriever = db.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        openai_api_base=vllm_base_url,
        openai_api_key="EMPTY",
        model=model_name,
        temperature=temperature,
    )
    return RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type="stuff")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_json(text: str, fallback: dict) -> dict:
    """Extract first JSON object from LLM output."""
    try:
        start = text.find("{")
        end   = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
    except Exception:
        pass
    return fallback


def _obs_to_info_dict(obs: np.ndarray, step: int,
                      keys: Optional[List[str]] = None) -> dict:
    """Convert numpy observation to JSON-serialisable dict."""
    if keys is None:
        keys = [f"obs_{i}" for i in range(len(obs))]
    d = {"step": step}
    for i, k in enumerate(keys):
        d[k] = round(float(obs[i]), 2) if i < len(obs) else 0.0
    return d


# ──────────────────────────────────────────────────────────────────────────────
# LLM Cooling Agent
# ──────────────────────────────────────────────────────────────────────────────

class LLMCoolingAgent:
    """
    LLM-based cooling controller for RackCooling-v0 / DataCenter-v0.

    Queries a vLLM server + RAG at each control step.
    """

    OBS_KEYS = [
        "T_gpu_mean_C", "T_gpu_hotspot_C", "T_hbm_C", "T_vrm_C",
        "T_cool_in_C", "T_cool_out_C", "P_gpu_W",
        "rack_T_in_C", "rack_T_out_C", "rack_total_power_W",
    ]

    def __init__(self,
                 docs_path: str = "docs/h100_thermal_notes.txt",
                 vllm_base_url: str = "http://localhost:8000/v1",
                 model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                 system_prompt: Optional[str] = None):
        self.system_prompt = system_prompt or COOLING_SYSTEM_PROMPT
        self._qa_chain = None
        self._docs_path = docs_path
        self._vllm_url  = vllm_base_url
        self._model     = model_name
        self._step      = 0
        self._last_action = 0.5

    def _ensure_chain(self):
        if self._qa_chain is None:
            self._qa_chain = build_rag_chain(
                self._docs_path, self._vllm_url, self._model)

    def predict(self, obs, deterministic=False):
        self._ensure_chain()
        obs_dict = _obs_to_info_dict(obs, self._step, self.OBS_KEYS)
        query = (self.system_prompt + "\n\nCurrent state:\n" +
                 json.dumps(obs_dict, indent=2) +
                 "\n\nRespond with JSON only.")
        try:
            llm_out = self._qa_chain.run(query)
            result  = _parse_json(llm_out,
                                  {"action_norm": self._last_action})
            action  = float(np.clip(
                result.get("action_norm", self._last_action), 0.0, 1.0))
        except Exception as e:
            print(f"[LLMCoolingAgent] Warning: {e}")
            action = self._last_action

        self._last_action = action
        self._step += 1
        return np.array([action], dtype=np.float32), {"llm_raw": str(result)}

    def reset(self):
        self._step = 0
        self._last_action = 0.5


# ──────────────────────────────────────────────────────────────────────────────
# LLM Joint Agent
# ──────────────────────────────────────────────────────────────────────────────

class LLMJointAgent:
    """
    LLM-based agent for JointDCFlat-v0 (4 actions: 3 cooling + 1 scheduling).
    """

    def __init__(self,
                 docs_path: str = "docs/h100_thermal_notes.txt",
                 vllm_base_url: str = "http://localhost:8000/v1",
                 model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        self.system_prompt = JOINT_SYSTEM_PROMPT
        self._qa_chain = None
        self._docs_path = docs_path
        self._vllm_url  = vllm_base_url
        self._model     = model_name
        self._step      = 0
        self._last = {"rack_flow_norm": 0.5, "cdu_pump_norm": 0.5,
                       "tower_fan_norm": 0.3, "queue_index": 0}

    def _ensure_chain(self):
        if self._qa_chain is None:
            self._qa_chain = build_rag_chain(
                self._docs_path, self._vllm_url, self._model)

    def predict(self, obs, deterministic=False):
        self._ensure_chain()
        obs_dict = _obs_to_info_dict(obs, self._step)
        query = (self.system_prompt + "\n\nCurrent state:\n" +
                 json.dumps(obs_dict, indent=2) +
                 "\n\nRespond with JSON only.")
        try:
            llm_out = self._qa_chain.run(query)
            result  = _parse_json(llm_out, self._last)
        except Exception as e:
            print(f"[LLMJointAgent] Warning: {e}")
            result = self._last

        action = np.array([
            float(np.clip(result.get("rack_flow_norm", 0.5), 0, 1)),
            float(np.clip(result.get("cdu_pump_norm", 0.5), 0, 1)),
            float(np.clip(result.get("tower_fan_norm", 0.3), 0, 1)),
            float(np.clip(result.get("queue_index", 0) / 10.0, 0, 1)),
        ], dtype=np.float32)

        self._last = result
        self._step += 1
        return action, {"llm_raw": str(result)}

    def reset(self):
        self._step = 0
        self._last = {"rack_flow_norm": 0.5, "cdu_pump_norm": 0.5,
                       "tower_fan_norm": 0.3, "queue_index": 0}
