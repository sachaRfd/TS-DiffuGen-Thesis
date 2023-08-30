# Diffusion Models for Optimized Geometry Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)
[![Powered by RDKit](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAwsPGi+MyC9RAAAAQElEQVQI12NgQABGQUEBMENISUkRLKBsbGwEEhIyBgJFsICLC0iIUdnExcUZwnANQWfApKCK4doRBsKtQFgKAQC5Ww1JEHSEkAAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyMi0wMy0xMVQxNToyNjo0NyswMDowMDzr2J4AAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjItMDMtMTFUMTU6MjY6NDcrMDA6MDBNtmAiAAAAAElFTkSuQmCC)](https://www.rdkit.org/)
![Tests Status](https://github.com/schwallergroup/DiffSach/actions/workflows/flake8.yml/badge.svg)
![Tests Status](https://github.com/schwallergroup/DiffSach/actions/workflows/tests.yml/badge.svg)

This repository contains the code for implementing Sacha Raffaud's IRP project titled "Diffusion Models for Optimized Geometry Prediction".

## Background

### Transition State Optimization

In this project, transition state optimization involves generating accurate 3D representations of transition states. This is achieved by utilizing reactant and product coordinates along with atom types. Optionally, reaction graphs can also be used as input, currently available with the initial W93 Dataset.



<p align="center">
  <div style="display: flex; justify-content: center;">
    <div style="margin-right: 20px;">
      <img src="visualisations/gifs/sample_5_1x.gif" alt="example gif" width="500">
    </div>
    <div>
      <figure style="margin: 0;">
        <img src="visualisations/gifs/sample_4_1x.gif" alt="example gif" width="500">
        <figcaption style="text-align: center;"></figcaption>
      </figure>
    </div>
  </div>
</p>



<!-- <div style="display: flex; justify-content: center;">
  <div style="margin-right: 20px;">
    <img src="visualisations/gifs/sample_5_1x.gif" alt="example gif" width="500">
  </div>
  <div>
    <figure style="margin: 0;">
      <img src="visualisations/gifs/sample_4_1x.gif" alt="example gif" width="500">
      <figcaption style="text-align: center;"></figcaption>
    </figure>
  </div>
</div> -->


### Equivariant Graph Neural Networks (EGNN)

EGNNs are graph neural networks (GNN) that maintain equivariance to transformations. This means that the output of the network transforms in the same manner as the input when transformed prior to feeding it into the GNN.

<div>
  <img src="visualisations/denoised_example_good.png" alt="Example Denoising image">
</div>
### Equivariant Diffusion Models

After establishing a denoising framework, it can be applied iteratively in a diffusion model. In diffusion models, the process involves forward diffusion, where noise is incrementally added to the input sample until it conforms to an isotropic Gaussian distribution. This is followed by learnable backward diffusion, where noise is subtracted to reconstruct the original data. This iterative backward process enables high-quality data generation.

For detailed background and results, refer to Sacha's thesis.

## Project Objectives

1. Adapt an EGNN network for denoising.
2. Implement this network within an equivariant diffusion process.
3. Enhance reaction/product information with additional chemical context.
4. Conduct ablation studies on the necessity of product information.
5. Explore the supplementation/replacement of reaction and product information with reaction graphs.

## Datasets

Three main datasets were employed in this project for comprehensive comparisons: W93, TX1, and RGD1 datasets. All datasets underwent conformation generation via DFT.

### W93 Dataset - Elementary Reactions of Gas-Phase Compounds

- Initial reaction conformation dataset with transition states.
- Around 12,000 samples for foundational analysis.
- Used with the TS-Diff model.

### TX1 Dataset - Transition X Dataset

- Built upon W93 with re-optimized transition states.
- Represents an improved version of W93.
- Used with the OA-ReactDiff model.

### RGD1 Dataset - Reaction Graph Depth One Dataset

- New dataset with over 176,000 samples.
- Features multiple transition state conformations.
- Includes larger molecules and offers new insights.

The primary dataset used is W93, comprehensively tested with PyTest. The other datasets have limited tests due to large .H5 files.

## Repository Overview

The main source files are located in the `src` directory, containing the following essential files:

- `train_test.py`: Script for training and testing diffusion models.
- `lightning_setup.py`: PyTorch Lightning class for diffusion models.
- `evaluate_samples.py`: Script for evaluating generated samples.

Subdirectories within `src` include `Diffusion` and `EGNN`, housing appropriate backbones for respective models. All dataset classes and setup files are in the `data` directory.

All dataset classes and setup files are located in the `data` directory.


# Training and Testing with Configuration Files

To facilitate seamless model training and testing, all operations are conducted through configuration files. Below is a brief overview of the various parameters that can be utilized within each diffusion model:

- **`train_test`**: Choose between 'Train' or 'Test': This directive controls whether a diffusion model should be trained or tested.
- **`use_graph_in_model`**: Boolean: Determines whether a reaction graph should be integrated into the model.
- **`dataset_to_use`**: Choose between 'W93', 'TX1', or 'RGD1': Specifies the dataset for training/sampling.
- **`timesteps`**: int: Dictates the number of diffusion steps to employ.
- **`noise_schedule`**: Choose between 'sigmoid_2', 'sigmoid_5', or 'cosine': Designates the noise schedule to employ.
- **`remove_hydrogens`**: Boolean: Determines whether hydrogens should be included.
- **`random_rotations`**: Boolean: Dictates whether random rotations should be applied during training.
- **`augment_train_set`**: Boolean: Controls whether the training set should be augmented by replacing reactants with products.
- **`include_context`**: Choose between None, 'Nuclear_Charges', 'Activation_Energy', or 'Van_Der_Waals': Specifies the type of context to incorporate.
- **`remove_product`**: Boolean: Controls whether product coordinates are incorporated.
- **`lr`**: float: Learning rate control.
- **`epochs`**: int: Number of training epochs.
- **`batch_size`**: int: Batch size for processing.
- **`learning_rate_scheduler`**: Boolean: Determines whether a scheduler should be utilized for learning rate.
- **`model_name`**: str: Model name for identification with WandB during training.
- **`folder_name`**: str: Name of the folder within the `trained_models` directory.
- **`wandb_project_name`**: str: Name of the project in WandB.
- **`n_layers`**: int: Number of EGNN layers in the diffusion model.
- **`hidden_features`**: int: Size of the embedding for hidden node features.

Example configuration files are available in the `configs` directory.

## Code Descriptions and Adaptations

Each Python file and script includes a header at the top, providing information about its contents. Additionally, there's a reference to any adaptations made from previous codebases.


# Usage:

## Setting up the Package: 

1. Clone the repository

```shell
git clone https://github.com/schwallergroup/DiffSach.git
```
2. Navigate to the root repository:

```shell
cd TS-DiffuGen
```

3. Create the Conda environment:

```shell
conda env create -f environment.yml
```

4. Activate the environment: 

```shell
conda activate tsdiff
```

5. Create the package: 

```shell
python setup.py install
```

6. Download the datasets and enjoy the package!

## Setting up the datasets:

### Setting up the W93 Dataset: 

To set up the W93 dataset, follow these steps:

1. Download the compressed Tar file `wb97xd3.tar.gz` from the following link: [W93 Dataset Link](https://zenodo.org/record/3715478)
2. Place the downloaded file in the `data/Dataset_W93/data/w93_dataset/` directory.
3. Uncompress the .tar file into the TS directory using the following command: 

     ```shell
     tar -xvf Dataset_W93/data/w93_dataset/wb97xd3.tar.gz -C Dataset_W93/data/TS/
     ```

4. Run the `setup_dataset_files.py` script to process and organize the dataset using the following command: 

    ```
    python data/Dataset_W93/setup_dataset_files.py
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
    python data/Dataset_RGD1/parse_data.py
    ```


## Setup WandB in Your Environment

The training of diffusion models is enhanced with the integration of Weights and Biases (WandB). This enables real-time visualization of training and validation losses, as well as continuous monitoring of the training process. To set up WandB:

1. Ensure you have a valid WandB API key available.
2. Place the API key within the `wandb_setup.py` file located in the root directory.
3. Run the `wandb_setup.py` script.


    ```shell
    python wandb_setup.py
    ```


4. With this setup, WandB is configured within your environment, allowing you to proceed with training diffusion models.


## Training a new Diffusion Model:

### Setup WandB in your environment: 

1. Change the parameters in the `configs/train_diffusion.yml` configuration file. 
2. Run the following command to train a new diffusion model:

    ```
    python src/train_test.py --config configs/train_diffusion.yml
    ```

## Sampling from test set using a pre-trained Diffusion Model:

The pre-trained_graph model was trained with the following parameters: 
- Uses Reaction Graphs
- Does not use Product coordinates 
- 1,000 sampling steps
- 8 EGNN layers
- 64 hidden features
- Sigmoid_2 noise schedule

The pre-trained_simple model was trained with the following parameters: 
- TX1 Dataset
- Without Reaction Graphs
- Using Product coordinates in inference 
- 2,000 sampling steps
- 8 EGNN layers
- 64 hidden features
- Sigmoid_2 noise schedule

1. Run the following script with the chosen testing config file:
    ```
    python src/train_test.py --config configs/test_pre_trained_diffusion_simple.yml
    ```
    or
    ```
    python src/train_test.py --config configs/test_pre_trained_diffusion_with_graphs.yml
    ```

2. Samples from the testset will be generated within the chosen model's Samples directory. This should take around 2 hours for the whole testset.



## Sampling from test set using a  Diffusion Model:

1. Adapt the parameters in the `configs/test_diffusion.yml` to match those that you used during training. 
2. Run the following command to sample from the test set using your trained diffusion model:

    ```
    python src/train_test.py --config configs/test_diffusion.yml
    ```

### Evaluating Samples

Before proceeding, ensure that all generated samples are organized within a designated `Samples` directory. Subsequently, execute the evaluation script by providing the path to your samples:

The evaluation script calculates the COV (Coverage) and MAT (Matching) scores for the generated samples, utilizing thresholds of 0.1 and 0.2 Å. Formulas for these metrics can be found in Sacha's thesis.

```shell
python src/evaluate_samples PATH_TO_SAMPLES_DIRECTORY
```
This command will trigger the evaluation process and display the computed COV and MAT scores on the screen.



## Visualisation with PyMol

To utilize PyMol for visualization:

1. Ensure PyMol is installed on your desktop along with the appropriate license. You can download it from [this link](https://pymol.org/2/).
2. Place the PyMol script in the designated folder and execute it using the PyMol GUI.

# Testing of the Code in This Repository

Testing for this project has been conducted using the PyTest framework. Thorough testing has been performed on the W93 dataset. Testing for the other two datasets requires the download of `.h5` files.

All critical functions, classes, and methods from various scripts have been rigorously tested and are located in the `/tests` directory. These tests are expected to pass successfully as part of the repository's workflow. If you intend to run these tests on your local machine, execute the following command:


```shell
pytest tests/
```

## References

[^1]: Grambow, C. A., Pattanaik, L., & Green, W. H. (2020). "Reactants, products, and transition states of elementary chemical reactions based on quantum chemistry." *2020.* [Link](https://doi.org/10.1038/s41597-020-0460-4)

[^2]: Schreiner, M., Bhowmik, A., Vegge, T., Busk, J., & Winther, O. (2022). Transition1x - a dataset for building generalizable reactive machine learning potentials. Scientific Data, 9(1), 779. [Link](https://doi.org/10.1038/s41597-022-01870-w)

[^3]: Zhao, Q., Vaddadi, S. M., Woulfe, M., Ogunfowora, L. A., Garimella, S. S., Isayev, O., & Savoie, B. M. (2023). Comprehensive exploration of graphically defined reaction spaces. Scientific Data, 10(1), 145. [Link](https://doi.org/10.1038/s41597-023-02043-z)

<!-- [^4]: Satorras, V. G., Hoogeboom, E., & Welling, M. "E(n) Equivariant Graph Neural Networks." *February 2021.* [arXiv](https://arxiv.org/abs/2102.09844).

[^5]: Hoogeboom, E., Satorras, V. G., Vignac, C., & Welling, M. "Equivariant Diffusion for Molecule Generation in 3D." *March 2022.* [arXiv](https://arxiv.org/abs/2203.05541).

 -->
