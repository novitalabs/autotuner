LLM Autotuner
=============

.. image:: _static/logo.svg
   :width: 128px
   :align: center
   :alt: LLM Autotuner Logo

Automated parameter tuning for LLM inference engines (SGLang, vLLM).

**Key Features:**

- **Multiple Deployment Modes**: Docker, Local (direct GPU), Kubernetes/OME
- **Optimization Strategies**: Grid search, Random search, Bayesian optimization
- **SLO-Aware Scoring**: Exponential penalties for constraint violations
- **GPU Intelligent Scheduling**: Per-GPU efficiency metrics and resource pooling
- **Web UI**: React frontend with real-time monitoring
- **Agent Assistant**: LLM-powered assistant for task management

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting-started/quickstart
   getting-started/installation

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user-guide/docker-mode
   user-guide/kubernetes
   user-guide/presets
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Features

   features/bayesian-optimization
   features/slo-scoring
   features/gpu-tracking
   features/parallel-execution
   features/websocket

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   architecture/deployment
   architecture/roadmap

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
