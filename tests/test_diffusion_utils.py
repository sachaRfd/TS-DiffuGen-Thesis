from src.Diffusion.utils import (
    assert_mean_zero,
    sample_center_gravity_zero_gaussian_with_mask,
    random_rotation,
)
import torch
import pytest

# Variables:
example_array = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ],
    dtype=torch.float32,
)
example_mask = torch.tensor(
    [
        [1, 1, 0],
    ]
)

example_batched_array = torch.tensor(
    [
        [
            [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        ]
    ],
    dtype=torch.float32,
)


def test_mean_not_zero():
    with pytest.raises(AssertionError):
        assert_mean_zero(example_array)


def test_mean_IS_zero():
    assert_mean_zero(
        example_array
        - example_array.mean(
            dim=1,
            keepdim=True,
        )
    )


def test_sample_gauss_wrong_shape():
    with pytest.raises(AssertionError):
        _ = sample_center_gravity_zero_gaussian_with_mask(
            size=example_array.size(),
            device=example_array.device,
            node_mask=example_mask,
        )


def test_sample_gauss():
    sample = sample_center_gravity_zero_gaussian_with_mask(
        size=example_array.unsqueeze(0).size(),
        device=example_array.device,
        node_mask=example_mask,
    )
    assert torch.allclose(
        torch.zeros(1),
        sample.mean(),
        atol=1e-5,
    )


def test_random_rot():
    x = example_batched_array[:, :, -3:]
    h = example_batched_array[:, :, :-3]
    x_, h_ = random_rotation(x, h)

    # Test that the SAME rotation was done on TS, R and P
    # Test the size of the output tensors
    assert x_.shape == x.shape
    assert h_.shape == h.shape

    # Assert that OHE has not been changed in the H vector:
    assert torch.allclose(h[:, :, :-6], h_[:, :, :-6])

    # Assert that the same rotation was applied to each item in the batch
    assert torch.allclose(x_[0][0], x_[0][1])

    # Check if the rotated data is different from the input
    assert not torch.allclose(x, x_)
    assert not torch.allclose(h, h_)

    # Get rotation of X, h
    rot_x = x / x_
    rot_h_reactant = h[:, :, -6:-3] / h_[:, :, -6:-3]
    rot_h_product = h[:, :, -3:] / h_[:, :, -3:]

    assert torch.allclose(rot_x, rot_h_product)
    assert torch.allclose(rot_x, rot_h_reactant)


if __name__ == "__main__":
    test_mean_not_zero()
    test_mean_IS_zero()
    test_sample_gauss_wrong_shape()
    test_sample_gauss()
    test_random_rot()
