# Sacha Raffaud sachaRfd and acse-sr1022

from src.EGNN.dynamics import EGNN_dynamics_QM9
import torch
import pytest

"""
Script to test classes and methods in the Dynamics file
"""

# Variables:
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
fake_sample_context = torch.tensor(
    [
        [
            [
                1.0000,
                0.0000,
                0.0000,
                0.0000,
                10.0000,
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
                10.0000,
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
fake_node_mask = torch.tensor([[1.0, 1.0]])
fake_edge_mask = torch.tensor(
    [
        [
            [
                0.0,
                1.0,
            ],
            [
                1.0,
                0.0,
            ],
        ]
    ]
)


def test_setup():
    in_node_nf = 10 + 1  # Number of h features + t

    _ = EGNN_dynamics_QM9(
        in_node_nf=in_node_nf,
    )


def setup(include_context=False):
    in_node_nf = 10 + 1  # Number of h features + t

    if include_context:
        in_node_nf += 1

    denoising_model = EGNN_dynamics_QM9(
        in_node_nf=in_node_nf,
    )
    return denoising_model


def test_adjacency_matrix():
    denoising_model = setup()
    n_nodes = 10

    rows, cols = denoising_model.get_adj_matrix(
        n_nodes=n_nodes,
        batch_size=1,
        device="cpu",
    )
    assert rows.shape[0] == cols.shape[0] == n_nodes**2


def test_forward_no_context():
    denoising_model = setup()
    t = torch.tensor(1)

    # We are not interested in node-feature updates:
    _, x_coords = denoising_model._forward(
        t=t,
        xh=fake_sample,
        node_mask=fake_node_mask,
        edge_mask=fake_edge_mask,
    )

    # assert that the prediction has shape of 1, 2, 3:
    assert x_coords.shape == fake_sample[:, :, -3:].shape


def test_forward_with_context():
    denoising_model = setup(include_context=True)
    t = torch.tensor(1)

    # Will have runtime error as we do not have have the right
    # shaped inputed:
    with pytest.raises(RuntimeError):
        _, x_coords = denoising_model._forward(
            t=t,
            xh=fake_sample,
            node_mask=fake_node_mask,
            edge_mask=fake_edge_mask,
        )

    # Run with sample with context ;
    _, x_coords = denoising_model._forward(
        t=t,
        xh=fake_sample_context,
        node_mask=fake_node_mask,
        edge_mask=fake_edge_mask,
    )

    # assert that the prediction has shape of 1, 2, 3:
    assert x_coords.shape == fake_sample_context[:, :, -3:].shape


if __name__ == "__main__":
    print("Running tests on dynamics script")
    test_setup()
    test_adjacency_matrix()
    test_forward_no_context()
    test_forward_with_context()
