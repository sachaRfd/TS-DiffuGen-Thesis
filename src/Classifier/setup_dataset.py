import torch
import os
from tqdm import tqdm
from src.EGNN.utils import setup_device
from src.lightning_setup import LitDiffusionModel
from src.Diffusion.saving_sampling_functions import write_xyz_file, return_xyz
from src.Diffusion.equivariant_diffusion import get_node_features

"""

This script can be used to generate samples from the training dataset for training of a classification model: 


"""  # noqa


if __name__ == "__main__":
    print("Running script")

    dataset_to_use = "W93"
    in_node_nf = get_node_features()
    hidden_features = 64
    n_layers = 6
    device = setup_device()
    lr = None
    remove_hydrogens = False
    timesteps = 1_000
    noise_schedule = "sigmoid_2"
    random_rotations = False
    augment_train_set = False
    include_context = None
    learning_rate_schedule = False
    no_product = False
    batch_size = 1

    # Variables:
    lit_diff_model = LitDiffusionModel(
        dataset_to_use=dataset_to_use,
        in_node_nf=in_node_nf,
        hidden_features=hidden_features,
        n_layers=n_layers,
        device=device,
        lr=lr,
        remove_hydrogens=remove_hydrogens,
        test_sampling_number=None,
        save_samples=False,
        save_path=None,
        timesteps=timesteps,
        noise_schedule=noise_schedule,
        random_rotations=random_rotations,
        augment_train_set=augment_train_set,
        include_context=include_context,
        learning_rate_schedule=learning_rate_schedule,
        no_product=no_product,
        batch_size=batch_size,
    )

    train_dataset = lit_diff_model.train_dataset
    print(len(train_dataset))

    # Load the models:
    model_path_saved = "trained_models/Model_with_DMAE_loss_each_sample_of_bactch_and_mse_3000_epochs/Weights/weights.pth"  # noqa
    lit_diff_model.load_state_dict(torch.load(model_path_saved))

    # Iterate over all the samples of train dataset:
    path = "data/Dataset_generated_samples/Clean/"
    number_samples = 40
    for i in tqdm(range(len(train_dataset))):
        current_path = path + f"Reaction_{i}/"
        os.makedirs(current_path, exist_ok=True)

        # Load the dataset:
        true_reaction, node_mask = train_dataset[i]
        atom_ohe = true_reaction[:, :4]
        reactant = true_reaction[:, 4:7]
        product = true_reaction[:, 7:10]
        true_TS = true_reaction[:, 10:13]

        # Create correct format:
        reactant = torch.cat([atom_ohe, reactant], dim=1)
        product = torch.cat([atom_ohe, product], dim=1)
        true_TS = torch.cat([atom_ohe, true_TS], dim=1)

        # Return them in XYZ format:
        reactant = return_xyz(
            sample=[reactant],
            ohe_dictionary=lit_diff_model.dataset.ohe_dict,
        )
        product = return_xyz(
            sample=[product],
            ohe_dictionary=lit_diff_model.dataset.ohe_dict,
        )
        true_TS = return_xyz(
            sample=[true_TS],
            ohe_dictionary=lit_diff_model.dataset.ohe_dict,
        )

        # Write them to the file:
        reactant_name = "Reactant_geometry.xyz"
        product_name = "Product_geometry.xyz"
        ts_name = "TS_geometry.xyz"
        write_xyz_file(reactant, current_path + reactant_name)
        write_xyz_file(product, current_path + product_name)
        write_xyz_file(true_TS, current_path + ts_name)

        # Generate Samples and save them:
        example_h = true_reaction[:, :-3].unsqueeze(0)
        example_x = true_reaction[:, -3:].unsqueeze(0)
        example_node_mask = node_mask.unsqueeze(0)
        # Create edge_mask:
        edge_mask = example_node_mask.unsqueeze(
            1,
        ) * example_node_mask.unsqueeze(
            2,
        )
        diag_mask = (
            ~torch.eye(edge_mask.size(-1), device=edge_mask.device)
            .unsqueeze(0)
            .bool()  # noqa
        )
        diag_mask = diag_mask.expand(edge_mask.size())
        edge_mask *= diag_mask

        inflated_h = example_h.repeat(number_samples, 1, 1)
        node_mask = node_mask.repeat(number_samples, 1)
        edge_mask = edge_mask.repeat(number_samples, 1, 1)

        lit_diff_model.diffusion_model.eval()
        lit_diff_model.to(device)
        samples = lit_diff_model.diffusion_model.sample(
            inflated_h.to(device),
            number_samples,
            23,
            node_mask.to(device),
            edge_mask.to(device),
            context_size=0,
        )

        # Write the samples locally:
        for i in range(number_samples):
            sample_name = f"Sample_{i}.xyz"
            sample = samples[i]
            sample = torch.cat([atom_ohe.to(device), sample], dim=1)
            sample = return_xyz(
                sample=[sample],
                ohe_dictionary=lit_diff_model.dataset.ohe_dict,
            )
            write_xyz_file(data=sample, filename=current_path + sample_name)
