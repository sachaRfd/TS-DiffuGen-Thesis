""""
This script it to create samples including their intermediate steps for visualisation as a GIF:   # noqa
----------------------------------------------------------------------------------------------
"""
import os
import torch
from src.lightning_setup import LitDiffusionModel
from Diffusion.saving_sampling_functions import write_xyz_file, return_xyz


if __name__ == "__main__":
    print("Running Chain Sampling script")

    # Setup variables:
    device = "cpu"

    dataset_to_use = "RGD1"

    # Setup Hyper-paremetres:
    learning_rate_schedule = False
    random_rotations = False  # Part of Data Augmentation
    augment_train_set = False  # Also part of Data Augmentation
    remove_hydrogens = False
    include_context = False

    # If we do not include the product in the diffusoin step:
    no_product = False

    if remove_hydrogens:
        in_node_nf = 9 + 1  # To account for time and 1 less OHE
    else:
        in_node_nf = 10 + 1  # To account for time

    if include_context:
        in_node_nf += 1  # Add one for the size of context --> For now we just have the Nuclear Charge  # noqa

    if no_product:
        in_node_nf -= 3

    out_node = 3
    context_nf = 0
    n_dims = 3
    noise_schedule = "sigmoid_2"
    loss_type = "l2"
    timesteps = 1_000
    batch_size = 64
    n_layers = 8
    hidden_features = 64
    lr = 1e-4
    epochs = 1000
    test_sampling_number = 40
    save_samples = True

    # Setup models:
    save_path = "src/Diffusion/RGD1_dataset_weights/False_no_productRGD1_dataset_False_include_VAN_DER_WAAL_RADII_False_Random_rotations_False_augment_train_set_8_layers_64_hiddenfeatures_0.001_lr_sigmoid_2_1000_timesteps_256_batch_size_100_epochs_False_Rem_Hydrogens/Sample_chain_5/"  # noqa
    lit_diff_model = LitDiffusionModel(
        dataset_to_use=dataset_to_use,
        in_node_nf=in_node_nf,
        context_nf=context_nf,
        hidden_features=hidden_features,
        out_node=out_node,
        n_dims=n_dims,
        n_layers=n_layers,
        device=device,
        lr=lr,
        remove_hydrogens=remove_hydrogens,
        test_sampling_number=1,
        save_samples=False,
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

    # Load weights:
    model_path = "src/Diffusion/RGD1_dataset_weights/False_no_productRGD1_dataset_False_include_VAN_DER_WAAL_RADII_False_Random_rotations_False_augment_train_set_8_layers_64_hiddenfeatures_0.001_lr_sigmoid_2_1000_timesteps_256_batch_size_100_epochs_False_Rem_Hydrogens/Weights/weights.pth"  # noqa
    lit_diff_model.load_state_dict(torch.load(model_path))

    # Setup the sample:
    example_sample, example_node_mask = lit_diff_model.dataset[5012]
    example_h = example_sample[:, :-3].unsqueeze(0)
    example_x = example_sample[:, -3:].unsqueeze(0)
    example_node_mask = example_node_mask.unsqueeze(0)

    # Create edge_mask:
    edge_mask = example_node_mask.unsqueeze(1) * example_node_mask.unsqueeze(2)
    diag_mask = (
        ~torch.eye(edge_mask.size(-1), device=edge_mask.device)
        .unsqueeze(0)
        .bool()  # noqa
    )
    diag_mask = diag_mask.expand(edge_mask.size())
    edge_mask *= diag_mask

    # Create the chain sampling:
    number_samples = 1
    keep_frames = 200
    inflated_h = example_h.repeat(number_samples, 1, 1)

    if dataset_to_use == "RGD1":
        sample_chain = lit_diff_model.diffusion_model.sample_chain(
            inflated_h,
            number_samples,
            33,
            keep_frames=keep_frames,
            node_mask=example_node_mask,
            edge_mask=edge_mask,
        )

    else:
        sample_chain = lit_diff_model.diffusion_model.sample_chain(
            inflated_h,
            number_samples,
            23,
            keep_frames=keep_frames,
            node_mask=example_node_mask,
            edge_mask=edge_mask,
        )

    # Save the samples:
    # 1. Get the atom OHE:
    atom_ohe = sample_chain[0, :, :4].unsqueeze(0)

    # Iterate over each sample and save it to the sample_folder;
    for i in range(keep_frames):
        # get the created output:
        predicted_samples = sample_chain[i, :, -3:].unsqueeze(0)

        # Concatenate the atomic composition with the predicted sample:
        predicted_samples = torch.cat([atom_ohe, predicted_samples], dim=2)
        predicted_samples = return_xyz(
            predicted_samples, dataset=lit_diff_model.dataset
        )

        # Save the file in the example_folder and with the name its Iteration number:  # noqa
        file_name = f"sample_{i}"
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, f"{file_name}.xyz")

        # Write the file:
        write_xyz_file(predicted_samples, full_path)

    # Save true sample:
    true_sample = torch.cat([atom_ohe, example_x], dim=2)
    true_sample = return_xyz(true_sample, dataset=lit_diff_model.dataset)
    file_name = "true_sample"
    full_path = os.path.join(save_path, f"{file_name}.xyz")
    write_xyz_file(true_sample, full_path)

    # Save true reactant
    true_reactant = example_sample[:, 4:7].unsqueeze(0)
    true_reactant = torch.cat([atom_ohe, true_reactant], dim=2)
    true_reactant = return_xyz(true_reactant, dataset=lit_diff_model.dataset)
    file_name = "true_reactant"
    full_path = os.path.join(save_path, f"{file_name}.xyz")
    write_xyz_file(true_reactant, full_path)

    # Save true product:
    true_product = example_sample[:, 7:10].unsqueeze(0)
    true_product = torch.cat([atom_ohe, true_product], dim=2)
    true_product = return_xyz(true_product, dataset=lit_diff_model.dataset)
    file_name = "true_product"
    full_path = os.path.join(save_path, f"{file_name}.xyz")
    write_xyz_file(true_product, full_path)

    print("Finished creating the chain samples")
