# Diffusion Models for Optimised Geometry Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)
![Tests Status](https://github.com/schwallergroup/DiffSach/actions/workflows/flake8.yml/badge.svg)
![Tests Status](https://github.com/schwallergroup/DiffSach/actions/workflows/tests.yml/badge.svg)


This repository contains the code for implementing Sacha Raffaud's IRP project titled "Diffusion Models for Optimised Geometry Prediction".



## Background:

### Transition State optimisation

In the context of this project, transition state optimization entails producing precise 3D representations of transition states. This is accomplished by employing reactant and product coordinates in conjunction with atom type. The user can also choose to use reaction graphs as input - but this is only available with the initial W93 Dataset.



<div style="display: flex; justify-content: center;">
  <div style="margin-right: 20px;">
    <img src="visualisations/gifs/sample_5_1x.gif" alt="example gif">
  </div>
  <div>
    <figure style="margin: 0;">
      <img src="visualisations/gifs/sample_4_1x.gif" alt="example gif">
      <figcaption style="text-align: center;"></figcaption>
    </figure>
  </div>
</div>

### Equivariant Graph Neural Networks (EGNN)

EGNNs are graph neural networks (GNN) that are equivariant to transformations. This means that if the input is transformed prior to being fed into the GNN, the the output should also be transformed the same.

EGNNs can be used as a denoising framework by inputting a noisy transition state as well as reactant and product information. The EGNN can be trained to predict the noise that was added to the input transition state, as can be seen in the figure below:

<div align="center">
  <img src="visualisations/denoised_example_good.png" alt="Example Denoising image">
</div>

### Equivariant Diffusion Models

Once a suitable denoising framework is in place, it can be used iteratively in a diffusion framework. This process can be shown below, where the EGNN condition on a time embedding can predict the noise added at each intermediate step in the figure below:

WRITE MORE


## This Project:

1. Adapt an EGNN network so that is can be used in a denoising task.
2. Implement this network in an equivariant diffusion process.
3. Supplement reaction/product information with additional chemical context.
4. Perform ablation studies on if product information can be removed.
5. Supplementing/Replacing reaction and product information with reaction graphs.

## Datasets: 

2 main datasets were used in this project so that our results could be compared with previous reports (W93 and TX1 datasets). One last dataset was found towards the end of the project and was only used in a couple of experiments (RGD1 datasets). All datasets had their conformations generated by DFT.


## W93 Dataset - Elementary Reactions of Gas-Phase Compounds [^1]

- First reaction conformation dataset, which includes transition states.
- Contains approximately 12,000 samples, providing a foundational dataset.
- Utilized the TS-Diff model for analysis.

## TX1 Dataset - Transition X Dataset [^2]

- Built upon the W93 Dataset with the distinction of re-optimized transition states.
- Serves as a refined version of the original W93 Dataset.
- Utilized in the OA-ReactDiff model for advanced analysis.

## RGD1 Dataset - Reaction Graph Depth One Dataset [^3]

- A recently released dataset designed to broaden the scope of analysis.
- Encompasses over 176,000 samples, incorporating multiple transition state conformations.
- Notably includes larger molecules compared to previous datasets, offering new insights.


The main dataset used for this project was the W93 dataset and therefore that one is fully tested with PyTest. The others were also used but have minor-to-no tests due to the large .H5 files that have to be downloaded.


## Description of useful Scripts:



## Description of the parameters in the training/testing config files: 


### Other scripts



## Code Adaptations

1. Code for the EGNN was adapted from  [E(n) Equivariant Graph Neural Networks](https://github.com/vgsatorras/egnn) [^2]
2. Code for the Variational Diffusion was adapted from [ E(3) Equivariant Diffusion Model for Molecule Generation in 3D](https://github.com/ehoogeboom/e3_diffusion_for_molecules/tree/main) [^3]


# Usage:

## Running the Package: 

1. Clone the repository

``$ git clone https://github.com/schwallergroup/DiffSach.git``

2. Travel to the root repository: 

``$ cd DiffSach``

3. Create the Conda environment:

``$ conda env create -f environment.yml``

4. Activate the environment: 

``$ conda activate tsdiff``

5. Create the package: 

``$ python setup.py install``

6. Download the datasets and enjoy the package!

## Setting up the datasets:

### Setting up the W93 Dataset: 

To set up the W93 dataset, follow these steps:

1. Download the compressed Tar file `wb97xd3.tar.gz` from the following link: [W93 Dataset Link](https://zenodo.org/record/3715478)
2. Place the downloaded file in the `data/Dataset_W93/data/w93_dataset/` directory.
3. Uncompress the .tar file into the TS directory using the following command: 

     ```shell
     $ tar -xvf Dataset_W93/data/w93_dataset/wb97xd3.tar.gz -C Dataset_W93/data/TS/
     ```

4. Run the `setup_dataset_files.py` script to process and organize the dataset using the following command: 

    ```
    $ python data/Dataset_W93/setup_dataset_files.py
    ```

### Setting up the TX1 Dataset: 

To set up the TX1 dataset, follow these steps:

1. Download the Transition1x.h5 file from the following link: [TX1 Dataset Link](https://zenodo.org/record/3715478)
2. Place the file in the  `data/Dataset_TX1` directory


### Setting up the RGD1 Dataset:

To set up the RGD1 dataset, follow these steps:

1. Download the RGD1_CHNO.h5 file from the following link: [RGD1 Dataset Link](https://doi.org/10.6084/m9.figshare.21066901.v6)
2. Place the file in the  `data/Dataset_RGD1` directory
3. Run the `parse_data.py` script with the following command:

    ```
    $ python data/Dataset_RGD1/parse_data.py
    ```


## Training a new Diffusion Model:

1. Change the parameters in the `configs/train_diffusion.yml` configuration file. 
2. Run the following command to train a new diffusion model:

    ```
    $ python src/train_test.py --config configs/train_diffusion.yml
    ```


## Sampling from test set using a Diffusion Model:

1. Adapt the parameters in the `configs/test_diffusion.yml` to match those that you used during training. 
2. Run the following command to sample from the test set using your trained diffusion model:

    ```
    $ python src/train_test.py --config configs/test_diffusion.yml
    ```



## Visualisation with PyMol: 

1. Make sure to have PyMol installed on the Desktop with the appropriate license: [PyMol Download Link](https://pymol.org/2/).
2. Then place PyMol Script in the appropriate folder and run it from the PyMol GUI. 





# Testing of the code in this repository: 

Testing for this project was performed using PyTest framework. Only the W93 dataset was thorougly tested due to the other two necessitating the download of .h5 files.

The main function, classes and methods of all other scripts have been tested can be found in the `/tests` directory. These tests should be passing in this repository as a workflow, however if you would like to rerun these locally please run the following command: 

  ```
  $ pytest tests/
  ```

## References

[^1]: Grambow, C. A., Pattanaik, L., & Green, W. H. (2020). "Reactants, products, and transition states of elementary chemical reactions based on quantum chemistry." *2020.* [Link](https://doi.org/10.1038/s41597-020-0460-4)

[^2]: Schreiner, M., Bhowmik, A., Vegge, T., Busk, J., & Winther, O. (2022). Transition1x - a dataset for building generalizable reactive machine learning potentials. Scientific Data, 9(1), 779. [Link](https://doi.org/10.1038/s41597-022-01870-w)

[^3]: Zhao, Q., Vaddadi, S. M., Woulfe, M., Ogunfowora, L. A., Garimella, S. S., Isayev, O., & Savoie, B. M. (2023). Comprehensive exploration of graphically defined reaction spaces. Scientific Data, 10(1), 145. [Link](https://doi.org/10.1038/s41597-023-02043-z)

[^4]: Satorras, V. G., Hoogeboom, E., & Welling, M. "E(n) Equivariant Graph Neural Networks." *February 2021.* [arXiv](https://arxiv.org/abs/2102.09844).

[^5]: Hoogeboom, E., Satorras, V. G., Vignac, C., & Welling, M. "Equivariant Diffusion for Molecule Generation in 3D." *March 2022.* [arXiv](https://arxiv.org/abs/2203.05541).


