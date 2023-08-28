"""
This is the script to evaluate our diffusion model using python Lightning
-------------------------------------------------------------------------
"""
import pytorch_lightning as pl
import torch
from src.EGNN import dynamics

from src.lightning_setup import LitDiffusionModel, LitDiffusionModel_With_graph
from src.Diffusion.equivariant_diffusion import get_node_features


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

    # Use Graph Model or not?
    use_reaction_graph_model = True

    # Setup Hyper-paremetres:
    learning_rate_schedule = False
    random_rotations = False  # Part of Data Augmentation
    augment_train_set = False  # Also part of Data Augmentation
    remove_hydrogens = False
    include_context = None  # "Activation_Energy"

    # If we do not include the product in the diffusoin step:
    no_product = True
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
    batch_size = 64
    n_layers = 8
    hidden_features = 64
    lr = 1e-4
    epochs = 1000
    test_sampling_number = 10
    save_samples = True
    save_path = "src/Diffusion/W93TESTING_FAKE_dataset_weights/True_no_product_True_graph_model_W93_dataset_None_context_False_Random_rotations_False_augment_train_set_8_layers_64_hiddenfeatures_0.0001_lr_sigmoid_2_1000_timesteps_294_batch_size_2000_epochs_False_Rem_Hydrogens/Samples/"  # noqa

    if not use_reaction_graph_model:
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
    model_path = "src/Diffusion/W93TESTING_FAKE_dataset_weights/True_no_product_True_graph_model_W93_dataset_None_context_False_Random_rotations_False_augment_train_set_8_layers_64_hiddenfeatures_0.0001_lr_sigmoid_2_1000_timesteps_294_batch_size_2000_epochs_False_Rem_Hydrogens/Weights/weights.pth"  # noqa

    # Load the state dict into the model:
    lit_diff_model.load_state_dict(torch.load(model_path))

    # Create a trainer instance for testing
    test_model(lit_diff_model, logger=None)
