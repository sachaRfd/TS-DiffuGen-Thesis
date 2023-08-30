# Sacha Raffaud sachaRfd and acse-sr1022

from src.lightning_setup import LitDiffusionModel
from src.Diffusion.equivariant_diffusion import get_node_features
from data.Dataset_W93.setup_dataset_files import process_reactions
import pytest
import shutil

import pandas as pd

"""
Script to test the Pytorch lightning class

Will only be tested with the W93 Dataset as this was the dataset
we focussed on, as the others require mode downloads.
"""

example_dataframe = pd.read_csv(
    "data/Dataset_W93/data/w93_dataset/wb97xd3.csv", index_col=0
)[:10]
temp_dir = "data/Dataset_W93/example_data_for_testing"
temp_dir_clean_geo = (
    "data/Dataset_W93/example_data_for_testing/Clean_Geometries"  # noqa
)


def setup_dataset():
    # Setup the dataset
    process_reactions(example_dataframe, temp_dir)


def delete_dataset():
    # Delete the dataset files created:
    shutil.rmtree(temp_dir_clean_geo)


def test_diff_train_setup():
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
            test_sampling_number=1,
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
            pytest_time=pytest_time,
        )

        train_loader = lit_diff_model.train_dataloader()
        batch = next(iter(train_loader))

        sample, sample_node_mask = batch

        # The rest of the shapes are well known
        assert (
            sample.shape[0] == batch_size
        )  # Will only work for batchsize of 1 as the test dataset is so small (only 10 samples) # noqa
        assert sample.shape[1] == 16
        assert sample.shape[2] == 13

        assert sample_node_mask.shape[0] == batch_size
        assert sample_node_mask.shape[1] == 16

        # Test every function:
        loss = lit_diff_model.training_step(batch=batch, batch_idx=0)
        # assert that the loss tensor includes requires grad to true:
        assert loss.requires_grad

        # Same for Val but make sure Requires grad is false:
        val_loader = lit_diff_model.val_dataloader()
        batch = next(iter(val_loader))

        sample, sample_node_mask = batch

        # The rest of the shapes are well known
        assert sample.shape[0] == batch_size
        assert sample.shape[1] == 16
        assert sample.shape[2] == 13

        assert sample_node_mask.shape[0] == batch_size
        assert sample_node_mask.shape[1] == 16

        loss = lit_diff_model.validation_step(batch=batch, batch_idx=0)

    finally:
        delete_dataset()


def test_remove_hydrogens():
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
            test_sampling_number=1,
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
            pytest_time=pytest_time,
        )

        train_loader = lit_diff_model.train_dataloader()
        batch = next(iter(train_loader))

        sample, sample_node_mask = batch

        # The rest of the shapes are well known
        assert (
            sample.shape[0] == batch_size
        )  # Will only work for batchsize of 1 as the test dataset is so small (only 10 samples) # noqa
        assert sample.shape[1] == 6  # No hydrogens are included
        assert sample.shape[2] == 12  # No Longer 13 as 1 OHE should be removed

        assert sample_node_mask.shape[0] == batch_size
        assert sample_node_mask.shape[1] == 6

        # Same for Val but make sure Requires grad is false:
        val_loader = lit_diff_model.val_dataloader()
        batch = next(iter(val_loader))

        sample, sample_node_mask = batch

        # The rest of the shapes are well known
        assert sample.shape[0] == batch_size
        assert sample.shape[1] == 6
        assert sample.shape[2] == 12

        assert sample_node_mask.shape[0] == batch_size
        assert sample_node_mask.shape[1] == 6

    finally:
        delete_dataset()


def test_include_context():
    # Variables:
    dataset_to_use = "W93"
    learning_rate_schedule = False
    random_rotations = False  # Part of Data Augmentation
    augment_train_set = False  # Also part of Data Augmentation
    remove_hydrogens = False  # Only Possible with the W93 Dataset
    include_context = "Activation_Energy"  # Only Possible with the W93 Dataset # noqa
    pytest_time = True

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
            test_sampling_number=1,
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
            pytest_time=pytest_time,
        )

        train_loader = lit_diff_model.train_dataloader()
        batch = next(iter(train_loader))

        sample, sample_node_mask = batch

        # The rest of the shapes are well known
        assert (
            sample.shape[0] == batch_size
        )  # Will only work for batchsize of 1 as the test dataset is so small (only 10 samples) # noqa
        assert sample.shape[1] == 16  # No hydrogens are included
        assert sample.shape[2] == 14  # No Longer 13 as 1 OHE should be removed

        assert sample_node_mask.shape[0] == batch_size
        assert sample_node_mask.shape[1] == 16

        # Same for Val but make sure Requires grad is false:
        val_loader = lit_diff_model.val_dataloader()
        batch = next(iter(val_loader))

        sample, sample_node_mask = batch

        # The rest of the shapes are well known
        assert sample.shape[0] == batch_size
        assert sample.shape[1] == 16
        assert sample.shape[2] == 14

        assert sample_node_mask.shape[0] == batch_size
        assert sample_node_mask.shape[1] == 16

    finally:
        delete_dataset()


def test_remove_hydrogens_and_including_context():
    # Variables:
    dataset_to_use = "W93"
    learning_rate_schedule = False
    random_rotations = False  # Part of Data Augmentation
    augment_train_set = False  # Also part of Data Augmentation
    remove_hydrogens = True  # Only Possible with the W93 Dataset
    include_context = "Van_Der_Waals"  # Only Possible with the W93 Dataset # noqa
    pytest_time = True

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
            test_sampling_number=1,
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
            pytest_time=pytest_time,
        )

        train_loader = lit_diff_model.train_dataloader()
        batch = next(iter(train_loader))

        sample, sample_node_mask = batch

        # The rest of the shapes are well known
        assert (
            sample.shape[0] == batch_size
        )  # Will only work for batchsize of 1 as the test dataset is so small (only 10 samples) # noqa
        assert sample.shape[1] == 6  # No hydrogens are included
        assert (
            sample.shape[2] == 13
        )  # Now Longer 13 as 1 OHE should be removed but 1 added due to context     # noqa

        assert sample_node_mask.shape[0] == batch_size
        assert sample_node_mask.shape[1] == 6

        # Same for Val but make sure Requires grad is false:
        val_loader = lit_diff_model.val_dataloader()
        batch = next(iter(val_loader))

        sample, sample_node_mask = batch

        # The rest of the shapes are well known
        assert sample.shape[0] == batch_size
        assert sample.shape[1] == 6
        assert sample.shape[2] == 13

        assert sample_node_mask.shape[0] == batch_size
        assert sample_node_mask.shape[1] == 6

    finally:
        delete_dataset()


def test_no_product():
    # Variables:
    dataset_to_use = "W93"
    learning_rate_schedule = False
    random_rotations = False  # Part of Data Augmentation
    augment_train_set = False  # Also part of Data Augmentation
    remove_hydrogens = False  # Only Possible with the W93 Dataset
    include_context = (
        None  # "Van_Der_Waals"  # Only Possible with the W93 Dataset # noqa
    )
    pytest_time = True

    # If we do not include the product in the diffusion step:
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
            test_sampling_number=1,
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
            pytest_time=pytest_time,
        )

        # No Real way of checking if it works except running an inference:
        train_loader = lit_diff_model.train_dataloader()
        batch = next(iter(train_loader))

        loss = lit_diff_model.training_step(batch=batch, batch_idx=0)
        # assert that the loss tensor includes requires grad to true:
        assert loss.requires_grad

    finally:
        delete_dataset()


def test_augment_train():
    # Variables
    dataset_to_use = "W93"
    learning_rate_schedule = False
    random_rotations = False  # Part of Data Augmentation
    augment_train_set = True  # Also part of Data Augmentation
    remove_hydrogens = False  # Only Possible with the W93 Dataset
    include_context = (
        None  # "Activation_Energy"  # Only Possible with the W93 Dataset # noqa
    )
    pytest_time = True

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
        lit_diff_model_augmented = LitDiffusionModel(
            dataset_to_use=dataset_to_use,
            in_node_nf=in_node_nf,
            hidden_features=hidden_features,
            n_layers=n_layers,
            device="cpu",
            lr=lr,
            remove_hydrogens=remove_hydrogens,
            test_sampling_number=1,
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
            pytest_time=pytest_time,
        )
        lit_diff_model_NOT_augmented = LitDiffusionModel(
            dataset_to_use=dataset_to_use,
            in_node_nf=in_node_nf,
            hidden_features=hidden_features,
            n_layers=n_layers,
            device="cpu",
            lr=lr,
            remove_hydrogens=remove_hydrogens,
            test_sampling_number=1,
            save_samples=False,
            save_path=None,
            timesteps=timesteps,
            noise_schedule=noise_schedule,
            random_rotations=random_rotations,
            augment_train_set=False,
            include_context=include_context,
            learning_rate_schedule=learning_rate_schedule,
            no_product=no_product,
            batch_size=batch_size,
            pytest_time=pytest_time,
        )
        # Assert that the augmented dataset is twice as big:
        assert (
            len(lit_diff_model_augmented.train_dataset)
            == len(lit_diff_model_NOT_augmented.train_dataset) * 2
        )

    finally:
        delete_dataset()


def test_assertions():
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
        # Assert that if saving wants to happen,
        # a path has to be given:
        with pytest.raises(AssertionError):
            _ = LitDiffusionModel(
                dataset_to_use=dataset_to_use,
                in_node_nf=in_node_nf,
                hidden_features=hidden_features,
                n_layers=n_layers,
                device="cpu",
                lr=lr,
                remove_hydrogens=remove_hydrogens,
                test_sampling_number=1,
                save_samples=True,
                save_path=None,
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

        # Test that if no product is set then augmentaed
        # train set cannot be true
        with pytest.raises(AssertionError):
            _ = LitDiffusionModel(
                dataset_to_use=dataset_to_use,
                in_node_nf=in_node_nf,
                hidden_features=hidden_features,
                n_layers=n_layers,
                device="cpu",
                lr=lr,
                remove_hydrogens=remove_hydrogens,
                test_sampling_number=1,
                save_samples=False,
                save_path=None,
                timesteps=timesteps,
                noise_schedule=noise_schedule,
                random_rotations=random_rotations,
                augment_train_set=True,
                include_context=include_context,
                learning_rate_schedule=learning_rate_schedule,
                no_product=True,
                batch_size=batch_size,
                pytest_time=pytest_time,
            )

    finally:
        delete_dataset()


if __name__ == "__main__":
    print("Running script")
    test_diff_train_setup()
    test_assertions()
    test_augment_train()
    test_include_context()
    test_remove_hydrogens()
    test_remove_hydrogens_and_including_context()
    test_no_product()
