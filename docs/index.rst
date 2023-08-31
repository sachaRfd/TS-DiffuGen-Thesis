TS-DiffuGen: Diffusion Model for Reaction Transition State optimisation
=======================================================================


Introduction:
=============

This file contains simple documentation regarding the TS-DiffuGen package. 


Datasets:
=========

The dataset classes can be found within the data directory. 

Setting up file: 
----------------
.. automodule:: data.Dataset_W93.setup_dataset_files
   :members:


Simple W93 Dataset without Graphs: 
----------------------------------
.. automodule:: data.Dataset_W93.dataset_class
   :members:

W93 Dataset with Graphs:
------------------------

.. automodule:: data.Dataset_W93.dataset_reaction_graph
   :members:


TX1 Dataset:
------------

.. automodule:: data.Dataset_TX1.TX1_dataloader
   :members:

.. automodule:: data.Dataset_TX1.dataset_TX1_class
   :members:


RGD1 Dataset:
-------------

.. automodule:: data.Dataset_RGD1.parse_data
   :members:

.. automodule:: data.Dataset_RGD1.RGD1_dataset_class
   :members:





The Equivariant Graph Neural Network:
=====================================

Background about equivariance and invariance as well as the architecture of the model is described in Sacha's thesis. 

The main EGNN files used for the diffusion process are the dynamics files: 

Dynamics without Graphs:
------------------------

.. automodule:: src.EGNN.egnn
   :members:

.. automodule:: src.EGNN.dynamics
   :members:


Dynamics with Graphs:
------------------------

The main difference being that the EGNN takes extra inputs of edge attributes.

.. automodule:: src.EGNN.egnn_with_graph
   :members:

.. automodule:: src.EGNN.dynamics_with_graph
   :members:


Utility Functions: 
------------------

The Utility functions used in the EGNN process are outlined bellow: 

.. automodule:: src.EGNN.utils
   :members:





Example Functions and Classes from the Package include:
=======================================================
See :mod:`src.evaluate_samples` for details.

.. automodule:: src.evaluate_samples
   :members:

