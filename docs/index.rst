TS-DiffuGen: Diffusion Model for Reaction Transition State Optimization
=====================================================================

Introduction
------------

Welcome to the TS-DiffuGen documentation. This guide provides an overview of the TS-DiffuGen package.

Datasets
--------

Explore the dataset classes in the data directory.

Setting up the dataset files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: data.Dataset_W93.setup_dataset_files
   :members:

Simple W93 Dataset without Graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: data.Dataset_W93.dataset_class
   :members:

W93 Dataset with Graphs
~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: data.Dataset_W93.dataset_reaction_graph
   :members:

TX1 Dataset
~~~~~~~~~~~
.. automodule:: data.Dataset_TX1.TX1_dataloader
   :members:
.. automodule:: data.Dataset_TX1.dataset_TX1_class
   :members:

RGD1 Dataset
~~~~~~~~~~~~
.. automodule:: data.Dataset_RGD1.parse_data
   :members:
.. automodule:: data.Dataset_RGD1.RGD1_dataset_class
   :members:

The Equivariant Graph Neural Network
------------------------------------

Learn about equivariance, invariance, and the model architecture in Sacha's thesis.

Dynamics without Graphs
~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: src.EGNN.egnn
   :members:
.. automodule:: src.EGNN.dynamics
   :members:

Dynamics with Graphs
~~~~~~~~~~~~~~~~~~~~~
Incorporating edge attributes into EGNN.

.. automodule:: src.EGNN.egnn_with_graph
   :members:
.. automodule:: src.EGNN.dynamics_with_graph
   :members:

Utility Functions
-----------------

Useful functions for the EGNN process.

.. automodule:: src.EGNN.utils
   :members:

The Diffusion Model Backbone
-----------------------------

Divided into two classes: one with edge attributes and one without.

.. automodule:: src.Diffusion.equivariant_diffusion
   :members:

Noise Schedule for Diffusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Separate class defining noise schedule for modularity.

.. automodule:: src.Diffusion.noising
   :members:

Utility Functions for Diffusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Shared utility functions for diffusion classes.

.. automodule:: src.Diffusion.utils
   :members:

PyTorch Lightning for Simplicity
---------------------------------

Integration of diffusion models, training, testing, and sampling into PyTorch Lightning class.
GPU recommended due to compute intensity.

Simple diffusion and graph-based diffusion classes

.. automodule:: src.lightning_setup
   :members:

Saving and Writing Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Functions for saving and writing.

.. automodule:: src.Diffusion.saving_sampling_functions
   :members:

Training and Sampling from the Test Set
---------------------------------------

Facilitated by config files in the configs directory.
Main function for training and sampling from the test set.

.. automodule:: src.train_test
   :members:

Evaluating Samples
------------------

Metrics COV (Coverage) and MAT (Matching) scores for molecule and conformation generation.

.. automodule:: src.evaluate_samples
   :members:

Supplementary Scripts and Functions
-----------------------------------

Includes `sample_diffusion_chain` script and other supplementary scripts.

Additional scripts in the Extra_scripts directory.

Note: Some scripts are for specific purposes like generating GIFs or saving molecule images in .XYZ format using PyMol.
