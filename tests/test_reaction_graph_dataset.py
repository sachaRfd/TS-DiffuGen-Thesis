from data.Dataset_W93.setup_dataset_files import process_reactions
from src.EGNN.utils import assert_mean_zero_with_mask

from data.Dataset_W93.dataset_reaction_graph import (
    W93_TS_coords_and_reacion_graph,
    get_adj_matrix_no_batch,
    bond_to_edge,
    get_bond_type,
)


import shutil

import torch
import pandas as pd
import pytest


# Example variables for dataset setup:
example_dataframe = pd.read_csv(
    "data/Dataset_W93/data/w93_dataset/wb97xd3.csv", index_col=0
)[:10]
temp_dir = "data/Dataset_W93/example_data_for_testing"
temp_dir_clean_geo = (
    "data/Dataset_W93/example_data_for_testing/Clean_Geometries"  # noqa
)


# Variables:

example_atom_1 = torch.tensor([[1, 0, 0, 0]])
example_atom_2 = torch.tensor([[1, 0, 0, 0]])
padding_atom = torch.tensor([[0, 0, 0, 0]])
wrong_atom = torch.tensor([[1, 1, 0, 0]])
example_distance = torch.tensor([0.4])


example_molecule = torch.tensor(
    [
        [
            1,
            0,
            0,
            0,
            1.2,
            3.2,
            1.9,
            0.2,
            1.2,
            2.1,
            1.1,
            1.6,
            0.7,
        ],
        [
            0,
            1,
            0,
            0,
            1.7,
            2.6,
            2.1,
            2.1,
            1.2,
            0.2,
            1.9,
            0.4,
            1.1,
        ],
    ]
)
reactant_bond = torch.tensor(
    [
        [0],
        [1],
        [1],
        [0],
    ]
)


def test_bond_type():
    bond_list = get_bond_type(example_atom_1, example_atom_2, example_distance)
    assert torch.isclose(torch.tensor(True), bond_list)


def test_padding_bond_type():
    value = get_bond_type(example_atom_1, padding_atom, example_distance)
    assert torch.isclose(torch.tensor(1), value)


def test_wrong_bond_type():
    with pytest.raises(ValueError):
        _ = get_bond_type(wrong_atom, example_atom_2, example_distance)


def test_adj_matrix():
    n_nodes = 23
    edge_index = get_adj_matrix_no_batch(n_nodes=n_nodes)
    assert len(edge_index) == 2
    assert edge_index[0].shape[0] == torch.tensor([23**2])
    assert edge_index[1].shape[0] == torch.tensor([23**2])


def test_get_bond_type():
    edge_index = get_adj_matrix_no_batch(n_nodes=2)
    h = example_molecule[:, :-3]
    reactant_bonds, _ = bond_to_edge(
        h=h,
        edge_index=edge_index,
    )
    assert torch.allclose(reactant_bonds, reactant_bond)


def setup_dataset():
    # Setup the dataset
    process_reactions(example_dataframe, temp_dir)
    return None


def delete_dataset():
    # Delete the dataset files created:
    shutil.rmtree(temp_dir_clean_geo)


def test_graph_dataset():
    # Setup the dataset:
    setup_dataset()
    dir = temp_dir_clean_geo
    try:
        dataset = W93_TS_coords_and_reacion_graph(
            directory=dir,
            running_pytest=True,
        )
        assert len(dataset) == 10
        sample, node_mask, edge_attr = dataset[0]

        assert sample.shape[0] == 16
        assert sample.shape[1] == 13

        assert node_mask.shape[0] == 16
        assert edge_attr.shape[0] == 16**2
        assert edge_attr.shape[1] == 2

        # Assert that the sample has CoM = 0
        for sample, node_mask, _ in dataset:
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
    finally:
        delete_dataset()


def test_wrong_graph_dataset():
    # Setup the dataset:
    setup_dataset()
    dir = temp_dir_clean_geo + "123"
    try:
        with pytest.raises(AssertionError):
            _ = W93_TS_coords_and_reacion_graph(
                directory=dir,
                running_pytest=True,
            )

    finally:
        delete_dataset()


if __name__ == "__main__":
    print("Running test script")
    # test_bond_type()
    # test_padding_bond_type()
    # test_wrong_bond_type()
    # test_get_bond_type()
    # test_adj_matrix()
    # test_graph_dataset()
    # test_wrong_graph_dataset()
