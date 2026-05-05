Citation
========

If you use **CompOpt** in your research, please cite the following:

.. code-block:: bibtex

   @inproceedings{compopt2026,
     title     = {CompOpt: An End-to-End Benchmark for Energy- and Water-Efficient
                  AI Data Center Control},
     author    = {},
     booktitle = {Advances in Neural Information Processing Systems (NeurIPS):
                  Datasets and Benchmarks Track},
     year      = {2026},
     url       = {https://github.com/your-org/compopt},
   }

Related Work
------------

This benchmark builds on several lines of research:

- **Data-center thermal modelling**: RC-network approximations of conjugate
  heat transfer, validated against CFD (see :doc:`guides/physics`).
- **Reinforcement learning for HVAC / cooling**: PPO and SAC policies for
  continuous control of coolant flow and CDU set-points.
- **LLM-based facility control**: Chain-of-thought reasoning over structured
  sensor observations for interpretable decision-making.
- **Workload-aware scheduling**: Co-optimisation of job placement and thermal
  management to reduce energy, water, and carbon footprint.

License
-------

CompOpt is released under the **MIT License**. See the ``LICENSE`` file in the
repository root for full terms.
