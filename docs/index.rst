TS-DiffuGen: Diffusion Model for Reaction Transition State Optimization
=====================================================================

Introduction
------------

Welcome to the TS-DiffuGen documentation. This guide provides an overview of the TS-DiffuGen package, which is designed for optimizing reaction transition states using diffusion models.

Datasets
--------

Explore the dataset classes in the data directory, which contain various datasets used for training and evaluation.

Setting up the dataset files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This section covers the process of setting up dataset files using the `Dataset_W93` module. It provides functions to prepare the necessary data for the models.
   
.. automodule:: data.Dataset_W93.setup_dataset_files
   :members:

Simple W93 Dataset without Graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The `Dataset_W93` module also includes a simple dataset class without graph information. This section details its functionalities.

.. automodule:: data.Dataset_W93.dataset_class
   :members:

W93 Dataset with Graphs
~~~~~~~~~~~~~~~~~~~~~~~~
For scenarios where graph structures are used in training, this section introduces the `Dataset_W93` module's dataset class that includes reaction graph information.

.. automodule:: data.Dataset_W93.dataset_reaction_graph
   :members:

TX1 Dataset
~~~~~~~~~~~
Explore the `Dataset_TX1` module for loading the TX1 dataset. It includes functionalities using a dataset classes that loads data from a dataloader.

.. automodule:: data.Dataset_TX1.TX1_dataloader
   :members:
.. automodule:: data.Dataset_TX1.dataset_TX1_class
   :members:

RGD1 Dataset
~~~~~~~~~~~~
Learn about the `Dataset_RGD1` module that provides functions to parse data for the RGD1 dataset. The associated dataset class is also explained here.

.. automodule:: data.Dataset_RGD1.parse_data
   :members:
.. automodule:: data.Dataset_RGD1.RGD1_dataset_class
   :members:

The Equivariant Graph Neural Network
------------------------------------

Theoretical background of equivariance, invariance, and the model architecture can be refered from Sacha's thesis. The EGNN architecture was used as a denoising framework. For this it was implemented within dynamics classes that wrap around the EGNN to make it suitable for a denoising task.

Dynamics without Graphs
~~~~~~~~~~~~~~~~~~~~~~~~
Learn about the core elements of the Equivariant Graph Neural Network (EGNN) without graph inputs.

.. automodule:: src.EGNN.egnn
   :members:
.. automodule:: src.EGNN.dynamics
   :members:

Dynamics with Graphs
~~~~~~~~~~~~~~~~~~~~~
This part focuses on incorporating edge attributes into EGNN's dynamics.

.. automodule:: src.EGNN.egnn_with_graph
   :members:
.. automodule:: src.EGNN.dynamics_with_graph
   :members:

Utility Functions
-----------------

The EGNN process involves several utility functions. This section provides insights into these functions.

.. automodule:: src.EGNN.utils
   :members:

The Diffusion Model Backbone
-----------------------------

This section describes the functions used for the equivariant diffusion model. One diffusion class includes edge attributes and another does not.

.. automodule:: src.Diffusion.equivariant_diffusion
   :members:

Noise Schedule for Diffusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Learn about the noise schedule used in the diffusion process. A separate class defines the noise schedule to enhance modularity.

.. automodule:: src.Diffusion.noising
   :members:

Utility Functions for Diffusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This section covers utility functions that are shared between the different diffusion classes, contributing to code modularity.

.. automodule:: src.Diffusion.utils
   :members:

PyTorch Lightning for Simplicity
---------------------------------

Explore how PyTorch Lightning is used to simplify the training, testing, and sampling processes. GPU usage is recommended due to the models' computational demands.

Simple diffusion and graph-based diffusion classes are explained here.

.. automodule:: src.lightning_setup
   :members:

Saving and Writing Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Learn about the functions dedicated to saving and writing different components of the diffusion models and their results.

.. automodule:: src.Diffusion.saving_sampling_functions
   :members:

Training and Sampling from the Test Set
---------------------------------------

This section covers the training and sampling procedures using config files located in the configs directory. Key functions are explained, offering insights into the process.

.. automodule:: src.train_test
   :members:

Evaluating Samples
------------------

Understand how sample evaluation is performed in the context of molecule and conformation generation. Metrics such as COV (Coverage) and MAT (Matching) are discussed in Sacha's thesis and their code implementations are defined below:

.. automodule:: src.evaluate_samples
   :members:

Supplementary Scripts and Functions
-----------------------------------

This section includes additional scripts and functions that supplement the main package functionalities. It highlights the role of scripts like `sample_diffusion_chain` and others in generating GIFs or saving molecule images in .XYZ format using PyMol.

Additional scripts can also be found in the Extra_scripts directory.
