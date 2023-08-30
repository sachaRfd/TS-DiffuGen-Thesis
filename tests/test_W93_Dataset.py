# Sacha Raffaud sachaRfd and acse-sr1022

from data.Dataset_W93.dataset_class import W93_TS
from data.Dataset_W93.setup_dataset_files import process_reactions
import pandas as pd
import shutil
import os
import torch
from src.EGNN.utils import assert_mean_zero_with_mask
from torch_geometric.data import Data


def test_dataset_wrong_path():
    dataframe = pd.read_csv(
        "data/Dataset_W93/data/w93_dataset/wb97xd3.csv",
        index_col=0,
    )[:10]
    directory = "data/Dataset_W93/example_data_for_testing"
    # First need to create the dataset:
    process_reactions(dataframe, directory)
    clean_geometries_directory = directory + "/Clean_Geometries"

    # Check that if wrong path is inputted then will get error:
    try:
        _ = W93_TS(directory=clean_geometries_directory + "test")
    except AssertionError:
        pass

    # Now we have to delete the files created by creating the dataset:
    reaction_directories = os.listdir(clean_geometries_directory)
    for directory in reaction_directories:
        directory_path = os.path.join(clean_geometries_directory, directory)  # noqa
        shutil.rmtree(directory_path)


def test_dataset_all_False():
    dataframe = pd.read_csv(
        "data/Dataset_W93/data/w93_dataset/wb97xd3.csv",
        index_col=0,
    )[:10]
    directory = "data/Dataset_W93/example_data_for_testing"
    # First need to create the dataset:
    process_reactions(dataframe, directory)

    # Check dataset works with no graph, no plot, with hydrogens, and without context:  # noqa
    clean_geometries_directory = directory + "/Clean_Geometries"
    dataset = W93_TS(
        directory=clean_geometries_directory,
        remove_hydrogens=False,
        graph=False,
        plot_distribution=False,
        include_context=False,
    )

    assert len(dataset) == 10  # as we only gave it 10 samples

    # assert ohe dictionary contains the correct atoms:
    assert dataset.ohe_dict == {
        "C": [1, 0, 0, 0],
        "N": [0, 1, 0, 0],
        "O": [0, 0, 1, 0],
        "H": [0, 0, 0, 1],
    }

    # Iterate over each sample and check that they have the same shape and can be used in dataloader:   # noqa
    for sample, node_mask in dataset:
        assert sample.shape == (16, 13)
        assert node_mask.shape[0] == 16

    # Now we have to delete the files created by creating the dataset:
    reaction_directories = os.listdir(clean_geometries_directory)
    for directory in reaction_directories:
        directory_path = os.path.join(clean_geometries_directory, directory)  # noqa
        shutil.rmtree(directory_path)


def test_dataset_no_hydrogen():
    dataframe = pd.read_csv(
        "data/Dataset_W93/data/w93_dataset/wb97xd3.csv",
        index_col=0,
    )[:10]
    directory = "data/Dataset_W93/example_data_for_testing"
    # First need to create the dataset:
    process_reactions(dataframe, directory)

    # Check dataset works with no graph, no plot, with hydrogens, and without context:  # noqa
    clean_geometries_directory = directory + "/Clean_Geometries"
    dataset = W93_TS(
        directory=clean_geometries_directory,
        remove_hydrogens=True,
        graph=False,
        plot_distribution=False,
        include_context=False,
    )

    assert len(dataset) == 10  # as we only gave it 10 samples

    # assert the ohe dictionary does not contain H
    assert dataset.ohe_dict == {"C": [1, 0, 0], "N": [0, 1, 0], "O": [0, 0, 1]}

    # Iterate over each sample and check that they have the same shape and can be used in dataloader:   # noqa
    for sample, node_mask in dataset:
        assert sample.shape == (6, 12)
        assert node_mask.shape[0] == 6

    # Now we have to delete the files created by creating the dataset:
    reaction_directories = os.listdir(clean_geometries_directory)
    for directory in reaction_directories:
        directory_path = os.path.join(clean_geometries_directory, directory)  # noqa
        shutil.rmtree(directory_path)


def test_dataset_graph():
    dataframe = pd.read_csv(
        "data/Dataset_W93/data/w93_dataset/wb97xd3.csv",
        index_col=0,
    )[:10]
    directory = "data/Dataset_W93/example_data_for_testing"
    # First need to create the dataset:
    process_reactions(dataframe, directory)

    # Check dataset works with no graph, no plot, with hydrogens, and without context:  # noqa
    clean_geometries_directory = directory + "/Clean_Geometries"
    dataset = W93_TS(
        directory=clean_geometries_directory,
        remove_hydrogens=False,
        graph=True,
        plot_distribution=False,
        include_context=False,
    )

    assert len(dataset) == 10  # as we only gave it 10 samples
    test_sample, _ = dataset[1]

    assert dataset.ohe_dict == {
        "C": [1, 0, 0, 0],
        "N": [0, 1, 0, 0],
        "O": [0, 0, 1, 0],
        "H": [0, 0, 0, 1],
    }

    # Check that the output is a geometric graph
    assert isinstance(test_sample, Data)

    # Now we have to delete the files created by creating the dataset:
    reaction_directories = os.listdir(clean_geometries_directory)
    for directory in reaction_directories:
        directory_path = os.path.join(clean_geometries_directory, directory)  # noqa
        shutil.rmtree(directory_path)


def test_dataset_plot():
    dataframe = pd.read_csv(
        "data/Dataset_W93/data/w93_dataset/wb97xd3.csv",
        index_col=0,
    )[:10]
    directory = "data/Dataset_W93/example_data_for_testing"
    # First need to create the dataset:
    process_reactions(dataframe, directory)

    # Check dataset works with no graph, no plot, with hydrogens, and without context:  # noqa
    clean_geometries_directory = directory + "/Clean_Geometries"
    dataset = W93_TS(
        directory=clean_geometries_directory,
        remove_hydrogens=False,
        graph=False,
        plot_distribution=True,
        include_context=False,
    )

    # assert that the image was created in the root directory:
    assert os.path.exists(dataset.path_to_save_image)
    os.remove(dataset.path_to_save_image)

    # Now we have to delete the files created by creating the dataset:
    reaction_directories = os.listdir(clean_geometries_directory)
    for directory in reaction_directories:
        directory_path = os.path.join(clean_geometries_directory, directory)  # noqa
        shutil.rmtree(directory_path)


def test_dataset_Wrong_Context():
    dataframe = pd.read_csv(
        "data/Dataset_W93/data/w93_dataset/wb97xd3.csv",
        index_col=0,
    )[:10]
    directory = "data/Dataset_W93/example_data_for_testing"
    # First need to create the dataset:
    process_reactions(dataframe, directory)

    # Check dataset works with no graph, no plot, with hydrogens, and without context:  # noqa
    clean_geometries_directory = directory + "/Clean_Geometries"
    try:
        _ = W93_TS(
            directory=clean_geometries_directory,
            remove_hydrogens=False,
            graph=False,
            plot_distribution=False,
            include_context="activation_Energy",
        )
    except AssertionError:
        pass

    # Now we have to delete the files created by creating the dataset:
    reaction_directories = os.listdir(clean_geometries_directory)
    for directory in reaction_directories:
        directory_path = os.path.join(clean_geometries_directory, directory)  # noqa
        shutil.rmtree(directory_path)


def test_dataset_AE_Context():
    dataframe = pd.read_csv(
        "data/Dataset_W93/data/w93_dataset/wb97xd3.csv",
        index_col=0,
    )[:10]
    directory = "data/Dataset_W93/example_data_for_testing"
    # First need to create the dataset:
    process_reactions(dataframe, directory)

    # Check dataset works with no graph, no plot, with hydrogens, and without context:  # noqa
    clean_geometries_directory = directory + "/Clean_Geometries"

    dataset = W93_TS(
        directory=clean_geometries_directory,
        remove_hydrogens=False,
        graph=False,
        plot_distribution=False,
        include_context="Activation_Energy",
    )

    # Only have 10 samples
    assert len(dataset) == 10

    assert dataset.ohe_dict == {
        "C": [1, 0, 0, 0],
        "N": [0, 1, 0, 0],
        "O": [0, 0, 1, 0],
        "H": [0, 0, 0, 1],
    }

    # Iterate over each sample and check that they have the same shape and can be used in dataloader:   # noqa
    for sample, node_mask in dataset:
        assert sample.shape == (16, 14)  # Plus 1
        assert node_mask.shape[0] == 16

    # Assert that the activation Energy is the same for each atom in the molecule:  # noqa
    for sample, _ in dataset:
        for atom in sample[1:]:
            assert atom[4:5] == sample[0, 4:5]

    # Assert that the activation energy is actually correct by comparing with the csv file: # noqa
    for index, (sample, _) in enumerate(dataset):
        assert torch.isclose(
            torch.tensor(
                dataframe.iloc[index].ea,
                dtype=torch.float32,
            ),
            sample[0, 4:5],
        )

    # Now we have to delete the files created by creating the dataset:
    reaction_directories = os.listdir(clean_geometries_directory)
    for directory in reaction_directories:
        directory_path = os.path.join(clean_geometries_directory, directory)  # noqa
        shutil.rmtree(directory_path)


# Make this test better by checking the OHE and Associated Nuclear Charge
def test_dataset_Nuclear_Charge_Context():
    dataframe = pd.read_csv(
        "data/Dataset_W93/data/w93_dataset/wb97xd3.csv",
        index_col=0,
    )[:10]
    directory = "data/Dataset_W93/example_data_for_testing"
    # First need to create the dataset:
    process_reactions(dataframe, directory)

    # Check dataset works with no graph, no plot, with hydrogens, and without context:  # noqa
    clean_geometries_directory = directory + "/Clean_Geometries"

    dataset = W93_TS(
        directory=clean_geometries_directory,
        remove_hydrogens=False,
        graph=False,
        plot_distribution=False,
        include_context="Nuclear_Charges",
    )

    # Only have 10 samples
    assert len(dataset) == 10

    # Iterate over each sample and check that they have the same shape and can be used in dataloader:   # noqa
    for sample, node_mask in dataset:
        assert sample.shape == (16, 14)  # Plus 1
        assert node_mask.shape[0] == 16

    # Now we have to delete the files created by creating the dataset:
    reaction_directories = os.listdir(clean_geometries_directory)
    for directory in reaction_directories:
        directory_path = os.path.join(clean_geometries_directory, directory)  # noqa
        shutil.rmtree(directory_path)


# Make it better
def test_dataset_Van_Der_Waals_Context():
    dataframe = pd.read_csv(
        "data/Dataset_W93/data/w93_dataset/wb97xd3.csv",
        index_col=0,
    )[:10]
    directory = "data/Dataset_W93/example_data_for_testing"
    # First need to create the dataset:
    process_reactions(dataframe, directory)

    # Check dataset works with no graph, no plot, with hydrogens, and without context:  # noqa
    clean_geometries_directory = directory + "/Clean_Geometries"

    dataset = W93_TS(
        directory=clean_geometries_directory,
        remove_hydrogens=False,
        graph=False,
        plot_distribution=False,
        include_context="Van_Der_Waals",
    )

    # Only have 10 samples
    assert len(dataset) == 10

    # Iterate over each sample and check that they have the same shape and can be used in dataloader:   # noqa
    for sample, node_mask in dataset:
        assert sample.shape == (16, 14)  # Plus 1
        assert node_mask.shape[0] == 16

    # Now we have to delete the files created by creating the dataset:
    reaction_directories = os.listdir(clean_geometries_directory)
    for directory in reaction_directories:
        directory_path = os.path.join(clean_geometries_directory, directory)
        shutil.rmtree(directory_path)


def test_dataset_centre_of_grav_is_zero():
    dataframe = pd.read_csv(
        "data/Dataset_W93/data/w93_dataset/wb97xd3.csv",
        index_col=0,
    )[:10]
    directory = "data/Dataset_W93/example_data_for_testing"
    # First need to create the dataset:
    process_reactions(dataframe, directory)

    # Check dataset works with no graph, no plot, with hydrogens, and without context:  # noqa
    clean_geometries_directory = directory + "/Clean_Geometries"

    dataset = W93_TS(
        directory=clean_geometries_directory,
        remove_hydrogens=False,
        graph=False,
        plot_distribution=False,
        include_context=None,
    )

    # Assert that the centre of Gravity of R, TS, and P is 0:
    for sample, node_mask in dataset:
        assert_mean_zero_with_mask(
            sample[:, -3:].reshape(1, dataset[0][0].shape[0], -1),
            node_mask=node_mask.reshape(1, dataset[0][0].shape[0], -1),
        )
        assert_mean_zero_with_mask(
            sample[:, -6:-3].reshape(1, dataset[0][0].shape[0], -1),
            node_mask=node_mask.reshape(1, dataset[0][0].shape[0], -1),
        )
        assert_mean_zero_with_mask(
            sample[:, -9:-6].reshape(1, dataset[0][0].shape[0], -1),
            node_mask=node_mask.reshape(1, dataset[0][0].shape[0], -1),
        )

    # Now we have to delete the files created by creating the dataset:
    reaction_directories = os.listdir(clean_geometries_directory)
    for directory in reaction_directories:
        directory_path = os.path.join(clean_geometries_directory, directory)
        shutil.rmtree(directory_path)


if __name__ == "__main__":
    print("Running Script")
    test_dataset_wrong_path()
    test_dataset_all_False()
    test_dataset_graph()
    test_dataset_no_hydrogen()
    test_dataset_plot()
    test_dataset_Wrong_Context()
    test_dataset_AE_Context()
    test_dataset_Nuclear_Charge_Context()
    test_dataset_Van_Der_Waals_Context()
    test_dataset_centre_of_grav_is_zero()
