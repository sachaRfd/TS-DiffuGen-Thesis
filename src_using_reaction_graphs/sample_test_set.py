import torch
import pytorch_lightning as pl

from Lightning_train import LitDiffusionModel

from src.EGNN.utils import setup_device

"""

This is the script to evaluate our diffusion model using python Lightning

"""


def test_model(model, logger):
    """
    Function to test the Lightning model
    """
    trainer = pl.Trainer(accelerator="cuda", logger=logger, fast_dev_run=False)
    trainer.test(model)


if __name__ == "__main__":
    device = setup_device()

    # Assign which dataset to use:
    dataset_to_use = "W93"
    use_product_graph = True
    no_product = True

    # Setup Hyper-paremetres:
    learning_rate_schedule = False
    random_rotations = False  # Part of Data Augmentation
    augment_train_set = False  # Also part of Data Augmentation
    remove_hydrogens = False  # Only Possible with the W93 Dataset
    include_context = False  # Only Possible with the W93 Dataset

    in_edge_nf = 2  # When we have the product in the graph

    if remove_hydrogens:
        in_node_nf = 9 + 1  # To account for time and 1 less OHE
    else:
        in_node_nf = 10 + 1  # To account for time

    if no_product:
        in_node_nf -= 6  # Was previously 3

    if include_context:
        in_node_nf += 1  # Add one for the size of context --> For now we just have the Nuclear Charge  # noqa

    out_node = 3
    context_nf = 0
    n_dims = 3
    noise_schedule = "sigmoid_2"
    timesteps = 1_000
    batch_size = 64
    n_layers = 8
    hidden_features = 64
    lr = 1e-4
    epochs = 2_000
    # sin_embedding = True
    test_sampling_number = 10
    save_samples = True
    save_path = "src_using_reaction_graphs/Diffusion/weights_and_samples/only_GRAPH_True_use_product_graph_W93_dataset_False_include_VAN_DER_WAAL_RADII_False_Random_rotations_False_augment_train_set_8_layers_64_hiddenfeatures_0.0001_lr_sigmoid_2_1000_timesteps_64_batch_size_5001_epochs_False_Rem_Hydrogens_second/Samples/"  # noqa

    # Setup model:
    lit_diff_model = LitDiffusionModel(
        dataset_to_use=dataset_to_use,
        in_node_nf=in_node_nf,
        context_nf=context_nf,
        in_edge_nf=in_edge_nf,
        hidden_features=hidden_features,
        out_node=out_node,
        n_dims=n_dims,
        n_layers=n_layers,
        device=device,
        lr=lr,
        remove_hydrogens=remove_hydrogens,
        test_sampling_number=test_sampling_number,
        save_samples=True,
        save_path=save_path,
        timesteps=timesteps,
        noise_schedule=noise_schedule,
        random_rotations=random_rotations,
        augment_train_set=augment_train_set,
        include_context=include_context,
        learning_rate_schedule=learning_rate_schedule,
        use_product_graph=use_product_graph,
        no_product=no_product,
    )

    print("Model parameters device:", next(lit_diff_model.parameters()).device)

    # Load the saved model state dictionary
    model_path = "src_using_reaction_graphs/Diffusion/weights_and_samples/only_GRAPH_True_use_product_graph_W93_dataset_False_include_VAN_DER_WAAL_RADII_False_Random_rotations_False_augment_train_set_8_layers_64_hiddenfeatures_0.0001_lr_sigmoid_2_1000_timesteps_64_batch_size_5001_epochs_False_Rem_Hydrogens_second/Weights/weights.pth"  # noqa
    lit_diff_model.load_state_dict(torch.load(model_path))

    # Create a trainer instance for testing
    answer = test_model(lit_diff_model, logger=None)

    # Second Run:

    save_path = "src_using_reaction_graphs/Diffusion/weights_and_samples/only_GRAPH_True_use_product_graph_W93_dataset_False_include_VAN_DER_WAAL_RADII_False_Random_rotations_False_augment_train_set_8_layers_64_hiddenfeatures_0.0001_lr_sigmoid_2_1000_timesteps_64_batch_size_5001_epochs_False_Rem_Hydrogens_second/Samples_2/"  # noqa

    # Setup model:
    lit_diff_model = LitDiffusionModel(
        dataset_to_use=dataset_to_use,
        in_node_nf=in_node_nf,
        context_nf=context_nf,
        in_edge_nf=in_edge_nf,
        hidden_features=hidden_features,
        out_node=out_node,
        n_dims=n_dims,
        n_layers=n_layers,
        device=device,
        lr=lr,
        remove_hydrogens=remove_hydrogens,
        test_sampling_number=test_sampling_number,
        save_samples=True,
        save_path=save_path,
        timesteps=timesteps,
        noise_schedule=noise_schedule,
        random_rotations=random_rotations,
        augment_train_set=augment_train_set,
        include_context=include_context,
        learning_rate_schedule=learning_rate_schedule,
        use_product_graph=use_product_graph,
        no_product=no_product,
    )

    print("Model parameters device:", next(lit_diff_model.parameters()).device)

    # Load the saved model state dictionary
    lit_diff_model.load_state_dict(torch.load(model_path))

    # Create a trainer instance for testing
    answer = test_model(lit_diff_model, logger=None)

    # Third Run:

    save_path = "src_using_reaction_graphs/Diffusion/weights_and_samples/only_GRAPH_True_use_product_graph_W93_dataset_False_include_VAN_DER_WAAL_RADII_False_Random_rotations_False_augment_train_set_8_layers_64_hiddenfeatures_0.0001_lr_sigmoid_2_1000_timesteps_64_batch_size_5001_epochs_False_Rem_Hydrogens_second/Samples_3/"  # noqa

    # Setup model:
    lit_diff_model = LitDiffusionModel(
        dataset_to_use=dataset_to_use,
        in_node_nf=in_node_nf,
        context_nf=context_nf,
        in_edge_nf=in_edge_nf,
        hidden_features=hidden_features,
        out_node=out_node,
        n_dims=n_dims,
        n_layers=n_layers,
        device=device,
        lr=lr,
        remove_hydrogens=remove_hydrogens,
        test_sampling_number=test_sampling_number,
        save_samples=True,
        save_path=save_path,
        timesteps=timesteps,
        noise_schedule=noise_schedule,
        random_rotations=random_rotations,
        augment_train_set=augment_train_set,
        include_context=include_context,
        learning_rate_schedule=learning_rate_schedule,
        use_product_graph=use_product_graph,
        no_product=no_product,
    )

    print("Model parameters device:", next(lit_diff_model.parameters()).device)

    # Load the saved model state dictionary
    lit_diff_model.load_state_dict(torch.load(model_path))

    # Create a trainer instance for testing
    answer = test_model(lit_diff_model, logger=None)
