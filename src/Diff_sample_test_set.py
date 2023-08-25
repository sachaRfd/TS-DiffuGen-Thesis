"""
This is the script to evaluate our diffusion model using python Lightning
-------------------------------------------------------------------------
"""
import pytorch_lightning as pl
import torch
from src.EGNN import dynamics

from src.Diff_train import LitDiffusionModel
from src.Diffusion.Equivariant_Diffusion import get_node_features


def test_model(model, logger):
    """
    Function to test the Lightning model
    """
    trainer = pl.Trainer(accelerator="cuda", logger=logger, fast_dev_run=False)
    trainer.test(model)


if __name__ == "__main__":
    # Hyper-parameters:
    device = dynamics.setup_device()

    dataset_to_use = "W93"

    # Setup Hyper-paremetres:
    learning_rate_schedule = False
    random_rotations = False  # Part of Data Augmentation
    augment_train_set = False  # Also part of Data Augmentation
    remove_hydrogens = False
    include_context = None  # "Activation_Energy"

    # If we do not include the product in the diffusoin step:
    no_product = False

    in_node_nf = get_node_features(
        remove_hydrogens=remove_hydrogens,
        include_context=include_context,
        no_product=no_product,
    )

    context_nf = 0
    noise_schedule = "sigmoid_2"
    loss_type = "l2"
    timesteps = 1_000
    batch_size = 64
    n_layers = 8
    hidden_features = 64
    lr = 1e-4
    epochs = 1000
    test_sampling_number = 10
    save_samples = True
    save_path = "src/Diffusion/W93_dataset_weights/False_no_product_W93_dataset_None_context_False_Random_rotations_True_augment_train_set_8_layers_64_hiddenfeatures_0.0001_lr_sigmoid_2_1000_timesteps_16_batch_size_1715_epochs_False_Rem_Hydrogens/Samples/"  # noqa

    # Create an instance of your Lightning model
    lit_diff_model = LitDiffusionModel(
        dataset_to_use=dataset_to_use,
        in_node_nf=in_node_nf,
        context_nf=context_nf,
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

    print("Model parameters device:", next(lit_diff_model.parameters()).device)

    # Load the saved model state dictionary
    model_path = "src/Diffusion/W93_dataset_weights/False_no_product_W93_dataset_None_context_False_Random_rotations_True_augment_train_set_8_layers_64_hiddenfeatures_0.0001_lr_sigmoid_2_1000_timesteps_16_batch_size_1715_epochs_False_Rem_Hydrogens/Weights/weights.pth"  # noqa

    # Load the state dict into the model:
    lit_diff_model.load_state_dict(torch.load(model_path))

    # Create a trainer instance for testing
    test_model(lit_diff_model, logger=None)

    # Second Run

    save_path = "src/Diffusion/W93_dataset_weights/False_no_product_W93_dataset_None_context_False_Random_rotations_True_augment_train_set_8_layers_64_hiddenfeatures_0.0001_lr_sigmoid_2_1000_timesteps_16_batch_size_1715_epochs_False_Rem_Hydrogens/Samples_2/"  # noqa

    # Create an instance of your Lightning model
    lit_diff_model = LitDiffusionModel(
        dataset_to_use=dataset_to_use,
        in_node_nf=in_node_nf,
        context_nf=context_nf,
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

    print("Model parameters device:", next(lit_diff_model.parameters()).device)

    # Load the state dict into the model:
    lit_diff_model.load_state_dict(torch.load(model_path))

    # Create a trainer instance for testing
    test_model(lit_diff_model, logger=None)

    # Third Run

    save_path = "src/Diffusion/W93_dataset_weights/False_no_product_W93_dataset_None_context_False_Random_rotations_True_augment_train_set_8_layers_64_hiddenfeatures_0.0001_lr_sigmoid_2_1000_timesteps_16_batch_size_1715_epochs_False_Rem_Hydrogens/Samples_3/"  # noqa

    # Create an instance of your Lightning model
    lit_diff_model = LitDiffusionModel(
        dataset_to_use=dataset_to_use,
        in_node_nf=in_node_nf,
        context_nf=context_nf,
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

    print("Model parameters device:", next(lit_diff_model.parameters()).device)

    # Load the state dict into the model:
    lit_diff_model.load_state_dict(torch.load(model_path))

    # Create a trainer instance for testing
    test_model(lit_diff_model, logger=None)
