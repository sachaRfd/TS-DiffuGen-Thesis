# Sacha Raffaud sachaRfd and acse-sr1022

from src.Diffusion.equivariant_diffusion import (
    get_node_features,
    DiffusionModel_graph,
)
from src.EGNN.dynamics_with_graph import EGNN_dynamics_graph

import pytest
import torch


"""
Testing all classes and methods of the diffusion graph class
"""


# Setup variables:
context_node_nf = 0
hidden_nf = 64
device = "cpu"
n_layers = 1

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
fake_edge_attributes = torch.tensor(
    [
        [
            0.0,
            1.0,
        ],
        [
            0.0,
            1.0,
        ],
        [
            0.0,
            1.0,
        ],
        [
            0.0,
            1.0,
        ],
    ]
)


def test_Diffusion_setup():
    # Setup internal Variables
    remove_hydrogen = False
    include_context = False

    # Setup Diffusion Variables:
    timesteps = 1_000
    correct_noise_schedule = "cosine"

    in_node_nf = get_node_features(
        remove_hydrogens=remove_hydrogen, include_context=include_context
    )

    in_edge_nf = 2
    # First setup EGNN model:
    denoising_model = EGNN_dynamics_graph(
        in_node_nf=in_node_nf,
        hidden_nf=hidden_nf,
        in_edge_nf=in_edge_nf,
        device=device,
        n_layers=n_layers,
    )

    # Setup Diffusion Model:
    _ = DiffusionModel_graph(
        dynamics=denoising_model,
        in_node_nf=in_node_nf,
        timesteps=timesteps,
        device=device,
        noise_schedule=correct_noise_schedule,
    )

    wrong_noise_schedule = "Cosine_123"
    with pytest.raises(ValueError):
        _ = DiffusionModel_graph(
            dynamics=denoising_model,
            in_node_nf=in_node_nf,
            timesteps=timesteps,
            device=device,
            noise_schedule=wrong_noise_schedule,
        )

    timestep_too_little = 10
    with pytest.raises(ValueError):
        _ = DiffusionModel_graph(
            dynamics=denoising_model,
            in_node_nf=in_node_nf,
            timesteps=timestep_too_little,
            device=device,
            noise_schedule=correct_noise_schedule,
        )


def setup():
    # Setup internal Variables
    remove_hydrogen = False
    include_context = False

    # Setup Diffusion Variables:
    timesteps = 1_000
    correct_noise_schedule = "cosine"

    in_node_nf = get_node_features(
        remove_hydrogens=remove_hydrogen,
        include_context=include_context,
    )
    in_edge_nf = 2

    # First setup EGNN model:
    denoising_model = EGNN_dynamics_graph(
        in_node_nf=in_node_nf,
        hidden_nf=hidden_nf,
        in_edge_nf=in_edge_nf,
        device=device,
        n_layers=n_layers,
    )

    # Setup Diffusion Model:
    diffusion_model = DiffusionModel_graph(
        dynamics=denoising_model,
        in_node_nf=in_node_nf,
        timesteps=timesteps,
        device=device,
        noise_schedule=correct_noise_schedule,
    )
    return diffusion_model


def test_diffusion_phi():
    # Setup diffusion model:
    diffusion_model = setup()

    # Setup fake_dataset:
    t = torch.tensor(10)

    # Setup size of the output:
    # Should be bs = 1, atoms = 2, dims = out_dims == 3
    shape_ = (1, 2, 3)

    phi_results = diffusion_model.phi(
        x=fake_sample,
        t=t,
        node_mask=fake_node_mask,
        edge_mask=fake_edge_mask,
        edge_attributes=fake_edge_attributes,
    )
    assert phi_results.shape == shape_


def test_sample():
    diffusion_model = setup()

    # setup variables
    h = fake_sample[:, :, :-3]
    n_sample = 1
    n_nodes = 2
    node_mask = fake_node_mask
    edge_mask = fake_edge_mask
    edge_attr = fake_edge_attributes

    # True shape of output should be 1 (bs) 2(number of nodes)
    # and then 3 (3d coordinates)
    true_shape = (n_sample, n_nodes, 3)

    output = diffusion_model.sample(
        h=h,
        n_samples=n_sample,
        n_nodes=n_nodes,
        node_mask=node_mask,
        edge_mask=edge_mask,
        edge_attributes=edge_attr,
    )
    assert output.shape == true_shape


if __name__ == "__main__":
    print("Running script")
    test_Diffusion_setup()
    test_diffusion_phi()
    test_sample()
