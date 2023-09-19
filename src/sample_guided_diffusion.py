from src.lightning_setup import LitDiffusionModel
from src.Diffusion.equivariant_diffusion import get_node_features

import numpy as np

from src.Classifier.classifier import EGNN_graph_prediction

from src.EGNN.egnn import get_adj_matrix

import torch

from src.evaluate_samples import calculate_distance_matrix, calculate_DMAE


from src.Diffusion.saving_sampling_functions import write_xyz_file, return_xyz  # noqa

"""

Example Script to use the guided diffusion

"""


def dont_use_hydrogens(input_tensor: torch.tensor) -> torch.tensor:
    """
    Dont use the row if it starts off with
    [0, 0, 1, 0] in the 4 first columns.
    """
    # Define the patterns to check for
    pattern1 = torch.tensor(
        [0, 0, 0, 1], dtype=input_tensor.dtype, device=input_tensor.device
    )
    pattern2 = torch.tensor(
        [0, 0, 0, 0], dtype=input_tensor.dtype, device=input_tensor.device
    )

    # Use torch.any to check if each row starts with either pattern
    mask1 = torch.all(input_tensor[:, :4] == pattern1, dim=1)
    mask2 = torch.all(input_tensor[:, :4] == pattern2, dim=1)

    # Combine the masks with logical OR to filter rows
    mask = ~(mask1 | mask2)

    # Use the combined mask to filter the tensor
    filtered_tensor = input_tensor[mask]

    return filtered_tensor


if __name__ == "__main__":
    print("Running script")

    # Load Pre-trained model:
    path_to_weights = "trained_models/Model_with_DMAE_loss_each_sample_of_bactch_and_mse_3000_epochs/Weights/weights.pth"  # noqa
    path_to_classifier = "trained_classifier/20_number_of_samples_64_bs_32_hidden_nf_1_layers_400_epoch_0.0002_lr/Weights/weights.pt"  # noqa

    # Setup the device:
    device = "cpu"

    # Variables used:
    use_mse = False
    use_graph_in_model = False
    dataset_to_use = "W93"
    timesteps = 1_000
    noise_schedule = "sigmoid_2"
    remove_hydrogens = False
    random_rotations = False
    augment_train_set = False
    include_context = None
    remove_product = False
    lr = 0.1
    epochs = 1
    learning_Rate_scheduler = False
    model_name = None
    n_layers = 6
    hidden_features = 64
    test_sampling_number = None

    in_node_nf = get_node_features(
        remove_hydrogens=remove_hydrogens,
    )

    diff_model = LitDiffusionModel(
        dataset_to_use=dataset_to_use,
        in_node_nf=in_node_nf,
        hidden_features=hidden_features,
        n_layers=n_layers,
        device=device,
        lr=lr,
        remove_hydrogens=remove_hydrogens,
        test_sampling_number=test_sampling_number,
        save_path=False,
        save_samples=False,
        timesteps=timesteps,
        noise_schedule=noise_schedule,
        random_rotations=random_rotations,
        augment_train_set=augment_train_set,
        include_context=include_context,
        learning_rate_schedule=learning_Rate_scheduler,
        no_product=False,
        batch_size=64,
        use_mse=False,
    )
    # Load diffusion weights:
    diff_model.load_state_dict(torch.load(path_to_weights))

    # Setup the Classifier model:
    classifier_model = EGNN_graph_prediction(
        in_node_nf=13,  # Have to include the TS coords     # noqa
        hidden_nf=32,
        out_node_nf=3,
        in_edge_nf=1,
        n_layers=1,
    )
    # Load the weights to the Classifier model:
    classifier_model.load_state_dict(torch.load(path_to_classifier))

    # Now we can look into the guiding functionality:
    dataset = diff_model.train_dataset

    sample, node_mask = dataset[21]

    # Unsqueeze to  see that it is as a batch
    sample_h = sample[:, :-3].unsqueeze(0)
    node_mask_example = node_mask.unsqueeze(0)

    # Setup the edge_mask:
    edge_mask = node_mask_example.unsqueeze(1) * node_mask_example.unsqueeze(
        2
    )  # noqa  # noqa
    diag_mask = (
        ~torch.eye(
            edge_mask.size(-1),
            device=edge_mask.device,
        )
        .unsqueeze(0)
        .bool()
    )
    diag_mask = diag_mask.expand(edge_mask.size())
    edge_mask *= diag_mask

    edge_mask_sample = edge_mask

    # Get the gudied sample
    number_of_samples = 5
    # Get adj matrix:
    edges = get_adj_matrix(
        n_nodes=23,
        batch_size=number_of_samples,
        device=device,
    )
    edges = [
        edge.to(device) for edge in edges
    ]  # Convert each tensor in the list to GPU tensor

    # Inflate the sizes to accomodate for the amount of samples
    # We want to generate
    inflated_h = sample_h.repeat(number_of_samples, 1, 1)
    node_mask_example = node_mask_example.repeat(number_of_samples, 1)
    edge_mask_sample = edge_mask_sample.repeat(number_of_samples, 1, 1)

    (
        test_guided_sample_true,
        test_guided_sample_list,
    ) = diff_model.diffusion_model.guided_sampling(
        h=inflated_h,
        n_samples=number_of_samples,
        n_nodes=23,
        node_mask=node_mask_example,
        edge_mask=edge_mask_sample,
        edges=edges,
        classifier_model=classifier_model,
    )

    # Remove the hydrogens from the true sample:
    sample_no_H = dont_use_hydrogens(sample)
    print(sample_no_H.shape)

    true_dm = calculate_distance_matrix(sample_no_H[:, -3:])

    for sample_ in test_guided_sample_list:
        list_dmae = []
        for sample_2 in sample_:
            # remove the hydrogens from the gen sample:
            gen_d = dont_use_hydrogens(sample_2)
            gen_d = calculate_distance_matrix(gen_d[:, -3:])
            res = calculate_DMAE(
                true_mol=true_dm,
                gen_mol=gen_d,
            )
            list_dmae.append(np.round(res, 2))
        print(list_dmae)

    list_dmae = []
    for i in range(number_of_samples):
        spl = test_guided_sample_true[i].clone().detach()
        spl = spl[: sample_no_H.shape[0], -3:]

        gen_d = calculate_distance_matrix(spl[:, -3:])
        res = calculate_DMAE(
            true_mol=true_dm,
            gen_mol=gen_d,
        )
        list_dmae.append(np.round(res, 2))

    print()
    print(list_dmae)

    # # Reverse the list:
    # # test_guided_sample_list.reverse()

    # # for count, item in enumerate(test_guided_sample_list):
    # #     xyz_format = return_xyz(
    # #         sample=torch.cat(
    # #             [sample_h[:, :, :4], item[:, :, -3:]],
    # #             dim=2,
    # #         ),
    # #         ohe_dictionary=dataset.ohe_dict,
    # #     )
    # #     write_xyz_file(xyz_format, f"guided_samples/generated_sample{count * 50}")    # noqa

    # # xyz_format_true = return_xyz(
    # #     sample=torch.cat(
    # #         [sample_h[:, :, :4], sample[:, -3:].unsqueeze(0)],
    # #         dim=2,
    # #     ),
    # #     ohe_dictionary=dataset.ohe_dict,
    # # )
    # # write_xyz_file(xyz_format_true, "guided_samples/True_sample")

    # # xyz_format = return_xyz(
    # #     sample=torch.cat(
    # #         [sample_h[:, :, :4], test_guided_sample_true[:, :, -3:]],
    # #         dim=2,
    # #     ),
    # #     ohe_dictionary=dataset.ohe_dict,
    # # )
    # # write_xyz_file(xyz_format, "guided_samples/generated_sample_zs")
    # # print("Done")
