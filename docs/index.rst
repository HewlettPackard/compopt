.. CompOpt documentation master file

CompOpt Documentation
=====================

**CompOpt** is an end-to-end simulator and benchmark for AI data-center control,
covering workload scheduling, compute budgeting, and energy/water/cost-efficient
liquid-cooling — all wrapped in `Gymnasium <https://gymnasium.farama.org/>`_
environments for reinforcement learning and agentic AI research.

Developed in collaboration with Anonymous, adapting concepts from the
`RAPS <https://exadigit.readthedocs.io/>`_ framework.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🚀 Getting Started
      :link: guides/quickstart
      :link-type: doc

      Install CompOpt and run your first simulation in under 5 minutes.

   .. grid-item-card:: 🌡️ Physics Engine
      :link: guides/physics
      :link-type: doc

      Chip → server → rack → datacenter thermal-fluid models.

   .. grid-item-card:: 🎮 Environments
      :link: guides/environments
      :link-type: doc

      Six Gymnasium environments at four difficulty levels.

   .. grid-item-card:: 🤖 RL Training
      :link: guides/rl_training
      :link-type: doc

      Train agents with Stable Baselines3, RLlib, or custom loops.

   .. grid-item-card:: 🧠 LLM Agents
      :link: guides/llm_agents
      :link-type: doc

      Agentic AI controllers using vLLM + LangChain RAG.

   .. grid-item-card:: 📖 API Reference
      :link: api/index
      :link-type: doc

      Full auto-generated API documentation.

Quick Example
-------------

.. code-block:: python

   import compopt
   from compopt.agents import PIDCoolingAgent

   env = compopt.make("RackCooling-v0")
   agent = PIDCoolingAgent(target_C=80.0, Kp=0.05)
   obs, info = env.reset()

   for _ in range(1800):
       action, _ = agent.predict(obs)
       obs, reward, terminated, truncated, info = env.step(action)
       if truncated:
           break

   print(f"Final hotspot: {info['T_hotspot_C']:.1f}°C")


.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   guides/quickstart
   guides/physics
   guides/environments
   guides/rewards
   guides/rl_training
   guides/llm_agents
   guides/configuration

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Project
   :hidden:

   changelog
   citation
