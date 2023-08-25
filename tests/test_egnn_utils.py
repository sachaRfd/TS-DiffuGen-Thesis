from src.EGNN.utils import (
    setup_device,
    sum_except_batch,
    remove_mean,
    remove_mean_with_mask,
    assert_mean_zero_with_mask,
    assert_correctly_masked,
)
import torch
import pytest
from unittest.mock import patch


"""
Script to test the utility functions for the equivariant graph neural network
"""

# Variables:

example_batch = torch.tensor(
    [[[1, 1, 1], [1, 1, 1]], [[2, 2, 2], [2, 2, 2]]], dtype=torch.float32
)

example_batch_list = torch.tensor(
    [
        [1, 1, 1],
        [2, 2, 2],
    ],
    dtype=torch.float32,
)

example_batch_padding = torch.tensor(
    [
        [9, 1, 3],
        [2, 5, 2],
    ],
    dtype=torch.float32,
)
example_batch_mask = torch.tensor([[1, 1, 0], [1, 1, 0]])


larger_example = torch.tensor(
    [[[2, 1, 2], [1, 2, 1]], [[3, 2, 3], [2, 3, 2]]], dtype=torch.float32
)


@pytest.mark.parametrize(
    "cuda_available, expected_device_type",
    [
        (False, "cpu"),  # Test with CUDA unavailable (CPU)
        (True, "cuda"),  # Test with CUDA available (GPU)
    ],
)
def test_setup_device(cuda_available, expected_device_type):
    with patch("torch.cuda.is_available", return_value=cuda_available):
        device = setup_device()
        assert device.type == expected_device_type


def test_sum_except_batch():
    sum = sum_except_batch(example_batch)
    assert torch.allclose(sum, torch.tensor([6.0, 12.0]))


def test_sum_except_batch_small_batch():
    sum = sum_except_batch(example_batch_list)
    assert torch.allclose(sum, torch.tensor([3.0, 6.0]))


def test_remove_mean():
    sample = remove_mean(example_batch_list)
    example_answer = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    assert torch.allclose(sample, example_answer)

    larger_example_answer = torch.tensor(
        [
            [[0.5, -0.5, 0.5], [-0.5, 0.5, -0.5]],
            [[0.5, -0.5, 0.5], [-0.5, 0.5, -0.5]],
        ]
    )

    larger_sample = remove_mean(larger_example)
    assert torch.allclose(larger_sample, larger_example_answer)


def test_remove_mean_mask():
    # First remove values that should be 0 at the mask
    example = example_batch_padding * example_batch_mask
    sample = remove_mean_with_mask(example, example_batch_mask)
    true_sample = torch.tensor([[4.0, -4.0, 0.0], [-1.5, 1.5, 0.0]])
    assert torch.allclose(sample, true_sample)


def test_correctly_masked_correct():
    assert_correctly_masked(
        example_batch_padding * example_batch_mask, example_batch_mask
    )


def test_correctly_masked_WRONG():
    with pytest.raises(AssertionError):
        assert_correctly_masked(example_batch_padding, example_batch_mask)


def test_mean_zero_with_mask_assertion_WRONG():
    # Not masked Properly
    with pytest.raises(AssertionError):
        assert_mean_zero_with_mask(example_batch_padding, example_batch_mask)

    # No mean 0:
    with pytest.raises(AssertionError):
        assert_mean_zero_with_mask(
            example_batch_padding * example_batch_mask,
            example_batch_mask,
        )


def test_mean_zero_with_mask_assertion_Correct():
    # Not masked Properly
    sample_mean_zero = remove_mean_with_mask(
        example_batch_padding * example_batch_mask,
        example_batch_mask,
    )
    assert_mean_zero_with_mask(
        sample_mean_zero,
        example_batch_mask,
    )


if __name__ == "__main__":
    print("Running testing for EGNN Utils")
    test_setup_device()
    test_sum_except_batch()
    test_sum_except_batch_small_batch()
    test_remove_mean()
    test_remove_mean_mask()
    test_correctly_masked_correct()
    test_correctly_masked_WRONG()
    test_mean_zero_with_mask_assertion_WRONG()
    test_mean_zero_with_mask_assertion_Correct()
