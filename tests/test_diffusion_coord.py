# Sacha Raffaud sachaRfd and acse-sr1022

from src.Diffusion.equivariant_diffusion import (
    get_node_features,
    DiffusionModel,
)
from src.EGNN.dynamics import EGNN_dynamics_QM9
from src.EGNN import utils as EGNN_utils

import pytest
import torch

"""
Testing all classes and methods of the diffusion script
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


def test_in_node_number():
    remove_hydrogens = False
    include_context = False

    simple_answer = 11
    simple_answer_2 = 12
    simple_answer_3 = 10

    in_node_simple = get_node_features(
        remove_hydrogens=remove_hydrogens, include_context=include_context
    )
    assert in_node_simple == simple_answer

    remove_hydrogens = False
    include_context = True
    in_node_simple_2 = get_node_features(
        remove_hydrogens=remove_hydrogens, include_context=include_context
    )
    assert in_node_simple_2 == simple_answer_2

    remove_hydrogens = True
    include_context = False
    in_node_simple_3 = get_node_features(
        remove_hydrogens=remove_hydrogens, include_context=include_context
    )
    assert in_node_simple_3 == simple_answer_3

    remove_hydrogens = True
    include_context = True
    in_node_simple_4 = get_node_features(
        remove_hydrogens=remove_hydrogens, include_context=include_context
    )
    assert in_node_simple_4 == simple_answer


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

    # First setup EGNN model:
    denoising_model = EGNN_dynamics_QM9(
        in_node_nf=in_node_nf,
        hidden_nf=hidden_nf,
        device=device,
        n_layers=n_layers,
    )

    # Setup Diffusion Model:
    _ = DiffusionModel(
        dynamics=denoising_model,
        in_node_nf=in_node_nf,
        timesteps=timesteps,
        device=device,
        noise_schedule=correct_noise_schedule,
    )

    wrong_noise_schedule = "Cosine_123"
    with pytest.raises(ValueError):
        _ = DiffusionModel(
            dynamics=denoising_model,
            in_node_nf=in_node_nf,
            timesteps=timesteps,
            device=device,
            noise_schedule=wrong_noise_schedule,
        )

    timestep_too_little = 10
    with pytest.raises(ValueError):
        _ = DiffusionModel(
            dynamics=denoising_model,
            in_node_nf=in_node_nf,
            timesteps=timestep_too_little,
            device=device,
            noise_schedule=correct_noise_schedule,
        )


def setup(include_context=False):
    # Setup internal Variables
    remove_hydrogen = False
    include_context = include_context

    # Setup Diffusion Variables:
    timesteps = 1_000
    correct_noise_schedule = "cosine"

    in_node_nf = get_node_features(
        remove_hydrogens=remove_hydrogen, include_context=include_context
    )

    # First setup EGNN model:
    denoising_model = EGNN_dynamics_QM9(
        in_node_nf=in_node_nf,
        hidden_nf=hidden_nf,
        device=device,
        n_layers=n_layers,
    )

    # Setup Diffusion Model:
    diffusion_model = DiffusionModel(
        dynamics=denoising_model,
        in_node_nf=in_node_nf,
        timesteps=timesteps,
        device=device,
        noise_schedule=correct_noise_schedule,
    )
    return diffusion_model


def test_diffusion_phi():
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

    # First setup EGNN model:
    denoising_model = EGNN_dynamics_QM9(
        in_node_nf=in_node_nf,
        hidden_nf=hidden_nf,
        device=device,
        n_layers=n_layers,
    )

    # Setup Diffusion Model:
    diffusion_model = DiffusionModel(
        dynamics=denoising_model,
        in_node_nf=in_node_nf,
        timesteps=timesteps,
        device=device,
        noise_schedule=correct_noise_schedule,
    )

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
    )
    assert phi_results.shape == shape_


def test_inflate_batch():
    batch_array = torch.tensor([1, 2, 3])
    target = torch.tensor([[0], [0], [0]])

    diffusion_model = setup()

    new_batch_array = diffusion_model.inflate_batch_array(
        array=batch_array, target=target
    )
    assert batch_array.shape != target.shape
    assert new_batch_array.shape == target.shape


def test_sample_position():
    diffusion_model = setup()

    n_samples = 1
    n_nodes = 10
    node_mask = torch.tensor([torch.ones(n_nodes).tolist()])

    output = diffusion_model.sample_position(
        n_samples=n_samples, n_nodes=n_nodes, node_mask=node_mask
    )

    # Test that mean has been removed
    EGNN_utils.assert_mean_zero_with_mask(
        output, node_mask.unsqueeze(2).expand(output.size())
    )


def test_sample_normal():
    diffusion_model = setup()

    mu = fake_sample[:, :, -3:]
    node_mask = fake_node_mask.expand(
        1,
        2,
    )
    zeros = torch.zeros(size=(mu.size(0), 1), device=mu.device)
    gamma_0 = diffusion_model.gamma(zeros)
    sigma = diffusion_model.SNR(-0.5 * gamma_0).unsqueeze(1)

    output = diffusion_model.sample_normal(
        mu=mu,
        sigma=sigma,
        node_mask=node_mask,
    )
    assert output.shape == mu.shape


def test_sample():
    diffusion_model = setup()

    # setup variables
    h = fake_sample[:, :, :-3]
    n_sample = 1
    n_nodes = 2
    node_mask = fake_node_mask
    edge_mask = fake_edge_mask

    # True shape of output should be 1 (bs) 2(number of nodes)
    # and then 3 (3d coordinates)
    true_shape = (n_sample, n_nodes, 3)

    output = diffusion_model.sample(
        h=h,
        n_samples=n_sample,
        n_nodes=n_nodes,
        node_mask=node_mask,
        edge_mask=edge_mask,
    )
    assert output.shape == true_shape


def test_sample_with_context():
    diffusion_model = setup(include_context=True)

    # setup variables
    h = fake_sample_context[:, :, :-3]
    n_sample = 1
    n_nodes = 2
    node_mask = fake_node_mask
    edge_mask = fake_edge_mask

    # True shape of output should be 1 (bs) 2(number of nodes)
    # and then 3 (3d coordinates)
    true_shape = (n_sample, n_nodes, 3)

    output = diffusion_model.sample(
        h=h,
        n_samples=n_sample,
        n_nodes=n_nodes,
        node_mask=node_mask,
        edge_mask=edge_mask,
        context_size=1,
    )

    with pytest.raises(AssertionError):
        _ = diffusion_model.sample(
            h=h,
            n_samples=n_sample,
            n_nodes=n_nodes,
            node_mask=node_mask,
            edge_mask=edge_mask,
            context_size=0,
        )

    assert output.shape == true_shape


def test_sample_chain():
    diffusion_model = setup()

    # setup variables
    h = fake_sample[:, :, :-3]
    n_sample = 1
    n_nodes = 2
    node_mask = fake_node_mask
    edge_mask = fake_edge_mask
    keep_frames = 100

    # True output shape should contain 100 frames
    true_shape = (keep_frames, n_nodes, 13)

    output = diffusion_model.sample_chain(
        h=h,
        n_samples=n_sample,
        n_nodes=n_nodes,
        node_mask=node_mask,
        edge_mask=edge_mask,
        keep_frames=keep_frames,
    )
    assert output.shape == true_shape


def test_sample_chain_with_context():
    diffusion_model = setup(include_context=True)

    # setup variables
    h = fake_sample_context[:, :, :-3]
    n_sample = 1
    n_nodes = 2
    node_mask = fake_node_mask
    edge_mask = fake_edge_mask
    keep_frames = 100

    # True output shape should contain 100 frames
    true_shape = (keep_frames, n_nodes, 14)

    with pytest.raises(AssertionError):
        _ = diffusion_model.sample_chain(
            h=h,
            n_samples=n_sample,
            n_nodes=n_nodes,
            node_mask=node_mask,
            edge_mask=edge_mask,
            keep_frames=keep_frames,
        )
    output = diffusion_model.sample_chain(
        h=h,
        n_samples=n_sample,
        n_nodes=n_nodes,
        node_mask=node_mask,
        edge_mask=edge_mask,
        keep_frames=keep_frames,
        context_size=1,
    )
    assert output.shape == true_shape


if __name__ == "__main__":
    print("Running script")
    test_in_node_number()
    test_Diffusion_setup()
    test_diffusion_phi()
    test_inflate_batch()
    test_sample_position()
    test_sample_normal()
    test_sample()
    test_sample_with_context()
    test_sample_chain()
    test_sample_chain_with_context()
