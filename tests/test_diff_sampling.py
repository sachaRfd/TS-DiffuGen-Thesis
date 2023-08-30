# Sacha Raffaud sachaRfd and acse-sr1022

from src.lightning_setup import LitDiffusionModel
import pandas as pd
from src.Diffusion.equivariant_diffusion import get_node_features
from data.Dataset_W93.setup_dataset_files import process_reactions
import os
import torch
import shutil

"""
Script to test the sampling functions of the diffusion process
"""


# Variables:
example_dataframe = pd.read_csv(
    "data/Dataset_W93/data/w93_dataset/wb97xd3.csv", index_col=0
)[:10]
temp_dir = "data/Dataset_W93/example_data_for_testing"
temp_dir_clean_geo = (
    "data/Dataset_W93/example_data_for_testing/Clean_Geometries"  # noqa
)
number_samples = 5
temp_path = "test_folder_sampling_123/"


def setup_dataset():
    # Setup the dataset
    process_reactions(example_dataframe, temp_dir)


def delete_dataset():
    # Delete the dataset files created:
    shutil.rmtree(temp_dir_clean_geo)


def test_sample_and_test_with_hydrogens():
    # Variables:
    dataset_to_use = "W93"
    learning_rate_schedule = False
    random_rotations = False  # Part of Data Augmentation
    augment_train_set = False  # Also part of Data Augmentation
    remove_hydrogens = False  # Only Possible with the W93 Dataset
    include_context = (
        None  # "Activation_Energy"  # Only Possible with the W93 Dataset # noqa
    )
    pytest_time = True

    # Create a temporary folder for samples:
    os.mkdir(temp_path)

    # If we do not include the product in the diffusoin step:
    no_product = False

    in_node_nf = get_node_features(
        remove_hydrogens=remove_hydrogens,
        include_context=include_context,
        no_product=no_product,
    )

    noise_schedule = "sigmoid_2"  # "sigmoid_INTEGER"
    timesteps = 1_000
    batch_size = 1
    n_layers = 1
    hidden_features = 10
    lr = 0.1

    # Setup the dataset:
    setup_dataset()

    try:
        lit_diff_model = LitDiffusionModel(
            dataset_to_use=dataset_to_use,
            in_node_nf=in_node_nf,
            hidden_features=hidden_features,
            n_layers=n_layers,
            device="cpu",
            lr=lr,
            remove_hydrogens=remove_hydrogens,
            test_sampling_number=number_samples,
            save_samples=True,
            save_path=temp_path,
            timesteps=timesteps,
            noise_schedule=noise_schedule,
            random_rotations=random_rotations,
            augment_train_set=augment_train_set,
            include_context=include_context,
            learning_rate_schedule=learning_rate_schedule,
            no_product=no_product,
            batch_size=batch_size,
            pytest_time=pytest_time,
        )

        test_loader = lit_diff_model.test_dataloader()
        batch = next(iter(test_loader))
        # Get sample and node_mask
        sample_example, node_mask_example = batch

        # Create edge mask:
        edge_mask_example = node_mask_example.unsqueeze(
            1
        ) * node_mask_example.unsqueeze(  # noqa
            2
        )  # noqa
        diag_mask = (
            ~torch.eye(
                edge_mask_example.size(-1),
                device=edge_mask_example.device,
            )
            .unsqueeze(0)
            .bool()
        )
        diag_mask = diag_mask.expand(edge_mask_example.size())
        edge_mask_example *= diag_mask

        # More variables:
        true_h = sample_example[0, :, :-3].unsqueeze(0)
        true_x = sample_example[0, :, -3:].unsqueeze(0)

        # test_sampling:
        lit_diff_model.sample_and_test(
            true_h=true_h,
            true_x=true_x,
            node_mask=node_mask_example,
            edge_mask=edge_mask_example,
            folder_path=temp_path,
            device=lit_diff_model.device,
        )

        # Assert that there are 5 files in the temp_dir
        list_temp_path = os.listdir(temp_path)
        assert len(list_temp_path) == 3 + number_samples

        # Assert that there are 3 files that start with true:
        true_files = [
            file for file in list_temp_path if file.startswith("true")  # noqa
        ]
        assert len(true_files) == 3

        # Asser that there are as many sample files as
        # the number of samples
        sample_files = [
            file for file in list_temp_path if file.startswith("sample")  # noqa
        ]
        assert len(sample_files) == number_samples
    finally:
        delete_dataset()
        shutil.rmtree(temp_path)


def test_sample_and_test_WITHOUT_hydrogens():
    # Variables:
    dataset_to_use = "W93"
    learning_rate_schedule = False
    random_rotations = False  # Part of Data Augmentation
    augment_train_set = False  # Also part of Data Augmentation
    remove_hydrogens = True  # Only Possible with the W93 Dataset
    include_context = (
        None  # "Activation_Energy"  # Only Possible with the W93 Dataset # noqa
    )
    pytest_time = True

    # Create a temporary folder for samples:
    os.mkdir(temp_path)

    # If we do not include the product in the diffusoin step:
    no_product = False

    in_node_nf = get_node_features(
        remove_hydrogens=remove_hydrogens,
        include_context=include_context,
        no_product=no_product,
    )

    noise_schedule = "sigmoid_2"  # "sigmoid_INTEGER"
    timesteps = 1_000
    batch_size = 1
    n_layers = 1
    hidden_features = 10
    lr = 0.1

    # Setup the dataset:
    setup_dataset()

    try:
        lit_diff_model = LitDiffusionModel(
            dataset_to_use=dataset_to_use,
            in_node_nf=in_node_nf,
            hidden_features=hidden_features,
            n_layers=n_layers,
            device="cpu",
            lr=lr,
            remove_hydrogens=remove_hydrogens,
            test_sampling_number=number_samples,
            save_samples=True,
            save_path=temp_path,
            timesteps=timesteps,
            noise_schedule=noise_schedule,
            random_rotations=random_rotations,
            augment_train_set=augment_train_set,
            include_context=include_context,
            learning_rate_schedule=learning_rate_schedule,
            no_product=no_product,
            batch_size=batch_size,
            pytest_time=pytest_time,
        )

        test_loader = lit_diff_model.test_dataloader()
        batch = next(iter(test_loader))
        # Get sample and node_mask
        sample_example, node_mask_example = batch

        # Create edge mask:
        edge_mask_example = node_mask_example.unsqueeze(
            1
        ) * node_mask_example.unsqueeze(  # noqa
            2
        )  # noqa
        diag_mask = (
            ~torch.eye(
                edge_mask_example.size(-1),
                device=edge_mask_example.device,
            )
            .unsqueeze(0)
            .bool()
        )
        diag_mask = diag_mask.expand(edge_mask_example.size())
        edge_mask_example *= diag_mask

        # More variables:
        true_h = sample_example[0, :, :-3].unsqueeze(0)
        true_x = sample_example[0, :, -3:].unsqueeze(0)

        # test_sampling:
        lit_diff_model.sample_and_test(
            true_h=true_h,
            true_x=true_x,
            node_mask=node_mask_example,
            edge_mask=edge_mask_example,
            folder_path=temp_path,
            device=lit_diff_model.device,
        )

        # Assert that there are 5 files in the temp_dir
        list_temp_path = os.listdir(temp_path)
        assert len(list_temp_path) == 3 + number_samples

        # Assert that there are 3 files that start with true:
        true_files = [
            file for file in list_temp_path if file.startswith("true")  # noqa
        ]
        assert len(true_files) == 3

        # Asser that there are as many sample files as
        # the number of samples
        sample_files = [
            file for file in list_temp_path if file.startswith("sample")  # noqa
        ]
        assert len(sample_files) == number_samples
    finally:
        delete_dataset()
        shutil.rmtree(temp_path)


def test_sample_and_test_NO_Product():
    # Variables:
    dataset_to_use = "W93"
    learning_rate_schedule = False
    random_rotations = False  # Part of Data Augmentation
    augment_train_set = False  # Also part of Data Augmentation
    remove_hydrogens = False  # Only Possible with the W93 Dataset
    include_context = (
        None  # "Activation_Energy"  # Only Possible with the W93 Dataset # noqa
    )
    pytest_time = True

    # Create a temporary folder for samples:
    os.mkdir(temp_path)

    # If we do not include the product in the diffusoin step:
    no_product = True

    in_node_nf = get_node_features(
        remove_hydrogens=remove_hydrogens,
        include_context=include_context,
        no_product=no_product,
    )

    noise_schedule = "sigmoid_2"  # "sigmoid_INTEGER"
    timesteps = 1_000
    batch_size = 1
    n_layers = 1
    hidden_features = 10
    lr = 0.1

    # Setup the dataset:
    setup_dataset()

    try:
        lit_diff_model = LitDiffusionModel(
            dataset_to_use=dataset_to_use,
            in_node_nf=in_node_nf,
            hidden_features=hidden_features,
            n_layers=n_layers,
            device="cpu",
            lr=lr,
            remove_hydrogens=remove_hydrogens,
            test_sampling_number=number_samples,
            save_samples=True,
            save_path=temp_path,
            timesteps=timesteps,
            noise_schedule=noise_schedule,
            random_rotations=random_rotations,
            augment_train_set=augment_train_set,
            include_context=include_context,
            learning_rate_schedule=learning_rate_schedule,
            no_product=no_product,
            batch_size=batch_size,
            pytest_time=pytest_time,
        )

        test_loader = lit_diff_model.test_dataloader()
        batch = next(iter(test_loader))
        # Get sample and node_mask
        sample_example, node_mask_example = batch

        # Create edge mask:
        edge_mask_example = node_mask_example.unsqueeze(
            1
        ) * node_mask_example.unsqueeze(  # noqa
            2
        )  # noqa
        diag_mask = (
            ~torch.eye(
                edge_mask_example.size(-1),
                device=edge_mask_example.device,
            )
            .unsqueeze(0)
            .bool()
        )
        diag_mask = diag_mask.expand(edge_mask_example.size())
        edge_mask_example *= diag_mask

        # More variables:
        # Remove the product from here
        true_h = sample_example[0, :, :-6].unsqueeze(0)
        true_x = sample_example[0, :, -3:].unsqueeze(0)

        # test_sampling:
        lit_diff_model.sample_and_test(
            true_h=true_h,
            true_x=true_x,
            node_mask=node_mask_example,
            edge_mask=edge_mask_example,
            folder_path=temp_path,
            device=lit_diff_model.device,
        )

        # Assert that there are 5 files in the temp_dir
        list_temp_path = os.listdir(temp_path)
        assert len(list_temp_path) == 2 + number_samples

        # Assert that there are 3 files that start with true:
        true_files = [
            file for file in list_temp_path if file.startswith("true")  # noqa
        ]
        assert len(true_files) == 2

        # Asser that there are as many sample files as
        # the number of samples
        sample_files = [
            file for file in list_temp_path if file.startswith("sample")  # noqa
        ]
        assert len(sample_files) == number_samples
    finally:
        delete_dataset()
        shutil.rmtree(temp_path)


def test_test_step():
    # Variables:
    dataset_to_use = "W93"
    learning_rate_schedule = False
    random_rotations = False  # Part of Data Augmentation
    augment_train_set = False  # Also part of Data Augmentation
    remove_hydrogens = False  # Only Possible with the W93 Dataset
    include_context = (
        None  # "Activation_Energy"  # Only Possible with the W93 Dataset # noqa
    )
    pytest_time = True

    # Create a temporary folder for samples:
    os.mkdir(temp_path)

    # If we do not include the product in the diffusoin step:
    no_product = False

    in_node_nf = get_node_features(
        remove_hydrogens=remove_hydrogens,
        include_context=include_context,
        no_product=no_product,
    )

    noise_schedule = "sigmoid_2"  # "sigmoid_INTEGER"
    timesteps = 1_000
    batch_size = 1
    n_layers = 1
    hidden_features = 10
    lr = 0.1

    # Setup the dataset:
    setup_dataset()

    try:
        lit_diff_model = LitDiffusionModel(
            dataset_to_use=dataset_to_use,
            in_node_nf=in_node_nf,
            hidden_features=hidden_features,
            n_layers=n_layers,
            device="cpu",
            lr=lr,
            remove_hydrogens=remove_hydrogens,
            test_sampling_number=number_samples,
            save_samples=True,
            save_path=temp_path,
            timesteps=timesteps,
            noise_schedule=noise_schedule,
            random_rotations=random_rotations,
            augment_train_set=augment_train_set,
            include_context=include_context,
            learning_rate_schedule=learning_rate_schedule,
            no_product=no_product,
            batch_size=batch_size,
            pytest_time=pytest_time,
        )

        # Get test loader and example batch:
        test_loader = lit_diff_model.test_dataloader()
        batch = next(iter(test_loader))

        # Run test function:
        lit_diff_model.test_step(batch=batch, batch_idx=0)

        path_to_test_batch = temp_path + "batch_0/mol_0"

        # Assert that there are 5 files in the temp_dir
        list_temp_path = os.listdir(path_to_test_batch)
        assert len(list_temp_path) == 3 + number_samples

        # Assert that there are 3 files that start with true:
        true_files = [
            file for file in list_temp_path if file.startswith("true")  # noqa
        ]
        assert len(true_files) == 3

        # Asser that there are as many sample files as
        # the number of samples
        sample_files = [
            file for file in list_temp_path if file.startswith("sample")  # noqa
        ]
        assert len(sample_files) == number_samples
    finally:
        delete_dataset()
        shutil.rmtree(temp_path)


if __name__ == "__main__":
    print("Running Script")
    test_sample_and_test_with_hydrogens()
    test_sample_and_test_WITHOUT_hydrogens()
    test_sample_and_test_NO_Product()
    test_test_step()
