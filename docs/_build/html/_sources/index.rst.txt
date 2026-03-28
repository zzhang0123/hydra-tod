hydra_tod Documentation
=======================

**Bayesian calibration and map-making for radio intensity mapping experiments**

``hydra_tod`` implements joint Bayesian inference of sky temperature maps,
instrument gains, and correlated noise parameters from radio telescope
time-ordered data (TOD). It uses Gibbs sampling with MPI parallelization
to efficiently handle multi-receiver, multi-scan datasets.

The package accompanies the paper:

    Zhang, Z., Bull, P., Santos, M. G., & Nasirudin, A. (2026).
    *Joint Bayesian calibration and map-making for intensity mapping experiments.*
    RAS Techniques and Instruments, rzag024.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Background

   theory/data_model
   theory/flicker_noise

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Project

   citation
   changelog


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
