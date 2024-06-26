# Sacha Raffaud sachaRfd and acse-sr1022

import pytorch_lightning as pl
import torch
from src.EGNN import dynamics

from src.lightning_setup import LitDiffusionModel, LitDiffusionModel_With_graph
from src.Diffusion.equivariant_diffusion import get_node_features


"""

This file can be used as a testing script for a trained diffusion model.
This is an older file and the testing/sampling can now be done directly using
the train_test.py file.

"""


def test_model(model, logger):
    """
    Function to test the Lightning model
    """
    trainer = pl.Trainer(accelerator="cuda", logger=logger, fast_dev_run=False)
    trainer.test(model)


if __name__ == "__main__":
    # Hyper-parameters:
    device = dynamics.setup_device()

    dataset_to_use = "RDKIT"

    # Use Graph Model or not?
    use_reaction_graph_model = False

    # Setup Hyper-paremetres:
    learning_rate_schedule = False
    random_rotations = False  # Part of Data Augmentation
    augment_train_set = False  # Also part of Data Augmentation
    remove_hydrogens = False
    include_context = None  # "Activation_Energy"

    # If we do not include the product in the diffusoin step:
    no_product = False
    in_edge_nf = 2  # When we have the product in the graph

    in_node_nf = get_node_features(
        remove_hydrogens=remove_hydrogens,
        include_context=include_context,
        no_product=no_product,
    )

    context_nf = 0
    noise_schedule = "sigmoid_2"
    loss_type = "l2"
    timesteps = 1_000
    batch_size = 128
    n_layers = 8
    hidden_features = 64
    lr = 1e-4
    epochs = 1000
    test_sampling_number = 10
    save_samples = True
    save_path = "trained_models/RDKIT/False_no_product_False_graph_model_RDKIT_dataset_None_context_False_Random_rotations_True_augment_train_set_8_layers_64_hiddenfeatures_5e-05_lr_sigmoid_2_1000_timesteps_128_batch_size_1000_epochs_False_Rem_Hydrogens/Samples/"  # noqa

    if not use_reaction_graph_model:
        # Create an instance of your Lightning model
        lit_diff_model = LitDiffusionModel(
            dataset_to_use=dataset_to_use,
            in_node_nf=in_node_nf,
            hidden_features=hidden_features,
            n_layers=n_layers,
            device=device,
            lr=lr,
            remove_hydrogens=remove_hydrogens,
            test_sampling_number=test_sampling_number,
            save_samples=save_samples,
            save_path=save_path,
            timesteps=timesteps,
            noise_schedule=noise_schedule,
            random_rotations=random_rotations,
            augment_train_set=augment_train_set,
            include_context=include_context,
            learning_rate_schedule=learning_rate_schedule,
            no_product=no_product,
            batch_size=batch_size,
        )
    else:
        # Setup Graph Diffusion model:
        lit_diff_model = LitDiffusionModel_With_graph(
            in_node_nf=in_node_nf,
            in_edge_nf=in_edge_nf,
            hidden_features=hidden_features,
            n_layers=n_layers,
            device=device,
            lr=lr,
            test_sampling_number=test_sampling_number,
            save_samples=True,
            save_path=save_path,
            timesteps=timesteps,
            noise_schedule=noise_schedule,
            learning_rate_schedule=learning_rate_schedule,
            no_product=no_product,
            batch_size=batch_size,
        )

    print("Model parameters device:", next(lit_diff_model.parameters()).device)

    # Load the saved model state dictionary
    model_path = "trained_models/RDKIT/False_no_product_False_graph_model_RDKIT_dataset_None_context_False_Random_rotations_True_augment_train_set_8_layers_64_hiddenfeatures_5e-05_lr_sigmoid_2_1000_timesteps_128_batch_size_1000_epochs_False_Rem_Hydrogens/Weights/weights.pth"  # noqa

    # Load the state dict into the model:
    lit_diff_model.load_state_dict(torch.load(model_path))

    # Create a trainer instance for testing
    test_model(lit_diff_model, logger=None)

    # Second Run:

    save_path = "trained_models/RDKIT/False_no_product_False_graph_model_RDKIT_dataset_None_context_False_Random_rotations_True_augment_train_set_8_layers_64_hiddenfeatures_5e-05_lr_sigmoid_2_1000_timesteps_128_batch_size_1000_epochs_False_Rem_Hydrogens/Samples_2/"  # noqa

    if not use_reaction_graph_model:
        # Create an instance of your Lightning model
        lit_diff_model = LitDiffusionModel(
            dataset_to_use=dataset_to_use,
            in_node_nf=in_node_nf,
            hidden_features=hidden_features,
            n_layers=n_layers,
            device=device,
            lr=lr,
            remove_hydrogens=remove_hydrogens,
            test_sampling_number=test_sampling_number,
            save_samples=save_samples,
            save_path=save_path,
            timesteps=timesteps,
            noise_schedule=noise_schedule,
            random_rotations=random_rotations,
            augment_train_set=augment_train_set,
            include_context=include_context,
            learning_rate_schedule=learning_rate_schedule,
            no_product=no_product,
            batch_size=batch_size,
        )
    else:
        # Setup Graph Diffusion model:
        lit_diff_model = LitDiffusionModel_With_graph(
            in_node_nf=in_node_nf,
            in_edge_nf=in_edge_nf,
            hidden_features=hidden_features,
            n_layers=n_layers,
            device=device,
            lr=lr,
            test_sampling_number=test_sampling_number,
            save_samples=True,
            save_path=save_path,
            timesteps=timesteps,
            noise_schedule=noise_schedule,
            learning_rate_schedule=learning_rate_schedule,
            no_product=no_product,
            batch_size=batch_size,
        )

    print("Model parameters device:", next(lit_diff_model.parameters()).device)

    # Load the state dict into the model:
    lit_diff_model.load_state_dict(torch.load(model_path))

    # Create a trainer instance for testing
    test_model(lit_diff_model, logger=None)

    # Third Run:

    save_path = "trained_models/RDKIT/False_no_product_False_graph_model_RDKIT_dataset_None_context_False_Random_rotations_True_augment_train_set_8_layers_64_hiddenfeatures_5e-05_lr_sigmoid_2_1000_timesteps_128_batch_size_1000_epochs_False_Rem_Hydrogens/Samples_3/"  # noqa

    if not use_reaction_graph_model:
        # Create an instance of your Lightning model
        lit_diff_model = LitDiffusionModel(
            dataset_to_use=dataset_to_use,
            in_node_nf=in_node_nf,
            hidden_features=hidden_features,
            n_layers=n_layers,
            device=device,
            lr=lr,
            remove_hydrogens=remove_hydrogens,
            test_sampling_number=test_sampling_number,
            save_samples=save_samples,
            save_path=save_path,
            timesteps=timesteps,
            noise_schedule=noise_schedule,
            random_rotations=random_rotations,
            augment_train_set=augment_train_set,
            include_context=include_context,
            learning_rate_schedule=learning_rate_schedule,
            no_product=no_product,
            batch_size=batch_size,
        )
    else:
        # Setup Graph Diffusion model:
        lit_diff_model = LitDiffusionModel_With_graph(
            in_node_nf=in_node_nf,
            in_edge_nf=in_edge_nf,
            hidden_features=hidden_features,
            n_layers=n_layers,
            device=device,
            lr=lr,
            test_sampling_number=test_sampling_number,
            save_samples=True,
            save_path=save_path,
            timesteps=timesteps,
            noise_schedule=noise_schedule,
            learning_rate_schedule=learning_rate_schedule,
            no_product=no_product,
            batch_size=batch_size,
        )

    print("Model parameters device:", next(lit_diff_model.parameters()).device)

    # Load the state dict into the model:
    lit_diff_model.load_state_dict(torch.load(model_path))

    # Create a trainer instance for testing
    test_model(lit_diff_model, logger=None)
