# Sacha Raffaud sachaRfd and acse-sr1022

"""

File to test the noising functions

"""
import pytest
import numpy as np
import torch
from src.Diffusion.noising import (
    clip_noise_schedule,
    polynomial_schedule,
    cosine_beta_schedule,
    sigmoid_beta_schedule,
    PredefinedNoiseSchedule,
    inflate_batch_array,
    sigma,
    alpha,
)


def test_clip_noise_schedule() -> None:
    alphas2 = np.array([0.1, 0.2, 0.3, 0.4])
    clipped_schedule = clip_noise_schedule(alphas2, clip_value=0.05)
    assert np.allclose(clipped_schedule, np.array([0.1, 0.1, 0.1, 0.1]))


def test_polynomial_schedule():
    timesteps = 10
    schedule = polynomial_schedule(timesteps)
    # Expected alpha^2 values for the default polynomial schedule
    expected_schedule = np.array(
        [
            9.99900000e-01,
            9.97901400e-01,
            9.83967187e-01,
            9.46639654e-01,
            8.76020781e-01,
            7.65571875e-01,
            6.14633069e-01,
            4.31662670e-01,
            2.38196371e-01,
            7.35263118e-02,
            1.73426312e-04,
        ]
    )

    assert np.allclose(schedule, expected_schedule)


def test_cosine_beta_schedule():
    timesteps = 10
    schedule = cosine_beta_schedule(timesteps, s=0.008, raise_to_power=1)

    expected_schedule = np.array(
        [
            9.76582333e-01,
            9.15167330e-01,
            8.20652335e-01,
            7.00574143e-01,
            5.64508006e-01,
            4.23304081e-01,
            2.88222224e-01,
            1.70034105e-01,
            7.81642577e-02,
            1.99385484e-02,
            1.99385484e-05,
        ]
    )

    assert np.allclose(schedule, expected_schedule)


def test_sigmoid_beta_schedule():
    timesteps = 10
    schedule = sigmoid_beta_schedule(timesteps, raise_to_power=1, S=6)

    expected_schedule = np.array(
        [
            0.99995222,
            0.99954179,
            0.99621953,
            0.97331805,
            0.86470704,
            0.59722353,
            0.29588894,
            0.11462605,
            0.03777662,
            0.00970549,
            0.001,
        ]
    )

    assert np.allclose(schedule, expected_schedule)


def test_inflate_batch_array():
    # Create an example batch array with shape (batch_size,)
    batch_array = torch.tensor([1, 2, 3, 4, 5])

    # Create a target tensor with the desired shape (batch_size, 1)
    target_tensor = torch.zeros((5, 1))

    # Inflate the batch array to match the target shape
    inflated_array = inflate_batch_array(batch_array, target_tensor)

    # Check if the inflated_array has the expected shape
    assert inflated_array.shape == (5, 1)

    # Check if the values of inflated_array are correct
    expected_output = torch.tensor([[1], [2], [3], [4], [5]])
    assert torch.all(torch.eq(inflated_array, expected_output))


def test_sigma():
    # Create an example gamma tensor
    gamma = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

    # Create a target tensor with the desired shape (batch_size, 1)
    target_tensor = torch.zeros((5, 1))

    # Compute sigma using the gamma tensor
    sigma_output = sigma(gamma, target_tensor)

    # Check if the sigma_output has the expected shape
    assert sigma_output.shape == (5, 1)

    # Check if the values of sigma_output are correct
    expected_output = torch.tensor(
        [[0.7246], [0.7415], [0.7579], [0.7737], [0.7890]]
    )  # noqa
    assert torch.allclose(sigma_output, expected_output, atol=1e-4)


def test_alpha():
    # Create an example gamma tensor
    gamma = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

    # Create a target tensor with the desired shape (batch_size, 1)
    target_tensor = torch.zeros((5, 1))

    # Compute alpha using the gamma tensor
    alpha_output = alpha(gamma, target_tensor)

    # Check if the alpha_output has the expected shape

    assert alpha_output.shape == (5, 1)

    # Check if the values of alpha_output are correct
    expected_output = torch.tensor(
        [[0.6892], [0.6709], [0.6523], [0.6335], [0.6144]]
    )  # noqa
    assert torch.allclose(alpha_output, expected_output, atol=1e-4)


def test_predefined_noise_schedule_invalid():
    timesteps = 10
    precision = 1e-4
    invalid_noise_schedule = "invalid_schedule"
    second_invalid = "polynomial_a"

    # Ensure that creating an instance with an invalid noise schedule raises a ValueError   # noqa
    with pytest.raises(ValueError):
        PredefinedNoiseSchedule(invalid_noise_schedule, timesteps, precision)
    with pytest.raises(ValueError):
        PredefinedNoiseSchedule(second_invalid, timesteps, precision)


if __name__ == "__main__":
    print("Running Tests on Noising script")
    pytest.main()
