# Sacha Raffaud sachaRfd and acse-sr1022

from src.EGNN.egnn import (
    coord2diff,
    unsorted_segment_sum,
    get_edges,
    get_edges_batch,
    EGNN,
)


import pytest
import torch

"""
Script to test classes and methods in the EGNN file
"""

# Example_variables:
# Fake dataset:
fake_sample = torch.tensor(
    [
        [
            [
                1.0000,
                0.0000,
                0.0000,
                0.0000,
                -1.3087,
                0.0068,
                0.0318,
                -1.0330,
                -0.0097,
                -0.0564,
                -1.1280,
                0.0562,
                0.0097,
            ],
            [
                1.0000,
                0.0000,
                0.0000,
                0.0000,
                1.3087,
                -0.0068,
                -0.0318,
                1.0330,
                0.0097,
                0.0564,
                -1.1280,
                0.0562,
                0.0097,
            ],
        ],
    ]
)


def test_edges():
    n_nodes = 100

    edges = get_edges(n_nodes=n_nodes)

    # assert len of edges is 2 for row and col
    assert len(edges) == 2

    # Because a node has no edge with itself
    assert len(edges[0]) == len(edges[1]) == (n_nodes**2 - n_nodes)


def test_get_edges():
    n_nodes = 10
    batch_size = 20

    edges, edge_attributes = get_edges_batch(
        n_nodes=n_nodes,
        batch_size=batch_size,
    )

    assert len(edges) == 2  # Should be 2 for row and col

    # Because a node has no edge with itself
    assert (
        len(edges[0])
        == len(edges[1])
        == (n_nodes * n_nodes * batch_size - (batch_size * n_nodes))
    )
    # Because a node has no edge with itself
    assert edge_attributes.shape == (
        (n_nodes * n_nodes * batch_size - (batch_size * n_nodes)),
        1,
    )


def test_coord2diff():
    x = fake_sample[0]
    edges = get_edges(
        n_nodes=2,
    )

    radial, coord_diff = coord2diff(x=x, edge_index=edges)

    # The two nodes input nodes are opposites of eachother:
    assert radial.shape[0] == fake_sample.shape[1]
    assert radial[0] == radial[1]

    # Coord diff should just be the same size as the input:
    assert coord_diff.shape == x.shape
    torch.allclose(
        coord_diff[:, -3:],
        torch.tensor(
            [[0, 0, 0], [0, 0, 0]],
            dtype=torch.float32,
        ),
    )


def test_unsorted_segment_sum():
    x = fake_sample[0]
    edges = get_edges(n_nodes=2)
    rows = edges
    rows = torch.tensor(rows[0])
    num_edges = fake_sample.size(1)

    # test wrong agg function:
    wrong_agg = "Mean"
    correct_agg = "mean"

    # Norm Factor:
    norm_factor = 100

    # Wrong agg function
    with pytest.raises(AssertionError):
        _ = unsorted_segment_sum(
            data=x,
            segment_ids=rows,
            num_segments=num_edges,
            normalization_factor=norm_factor,
            aggregation_method=wrong_agg,
        )

    agg = unsorted_segment_sum(
        data=x,
        segment_ids=rows,
        num_segments=num_edges,
        normalization_factor=norm_factor,
        aggregation_method=correct_agg,
    )

    assert torch.allclose(agg, x)

    # sum aggregation:
    correct_agg = "sum"

    agg = unsorted_segment_sum(
        data=x,
        segment_ids=rows,
        num_segments=num_edges,
        normalization_factor=norm_factor,
        aggregation_method=correct_agg,
    )

    # For our simple case the sum agg will just be the
    # initial data divided by the norm factor
    assert torch.allclose(agg, x / norm_factor)


def test_egnn_class_setup():
    # Dummy parameters
    _ = EGNN(
        in_node_nf=10,
        hidden_nf=64,
        in_edge_nf=1,
    )


if __name__ == "__main__":
    print("Running EGNN tests")
    test_edges()
    test_get_edges()
    test_coord2diff()
    test_unsorted_segment_sum()
    test_egnn_class_setup()
