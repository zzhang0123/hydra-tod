Tutorials
=========

These tutorials walk through common workflows using ``hydra_tod``.
Complete Jupyter notebooks are available in the ``examples/`` directory
of the repository.

Single-TOD Analysis
-------------------

The simplest use case: simulate a single telescope scan and run the
Gibbs sampler to recover sky, gain, and noise parameters.

See notebook: ``examples/workflow_GS1.ipynb``

Multi-TOD Analysis
------------------

Combine multiple scans (e.g., setting and rising) to jointly infer
shared sky parameters with per-scan gain and noise parameters.

See notebook: ``examples/workflow_TODs_GS5_linear.ipynb``

Flicker Noise Demonstration
----------------------------

Explore the properties of 1/f flicker noise: correlation functions,
power spectral densities, and the effect of spectral index and knee
frequency parameters.

See notebook: ``examples/flicker_noise_demonstration.ipynb``
