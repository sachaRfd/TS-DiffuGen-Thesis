import torch
import numpy as np

from Dataset_W93.dataset_class import *
from torch_geometric.loader import DataLoader


def sum_except_batch(x):
    """
    Sums the elements of each tensor in the input `x`, except the batch dimension.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, *).

    Returns:
        torch.Tensor: A tensor of shape (batch_size,) containing the sum of elements in each tensor of `x`, 
                      excluding the batch dimension.
    """
    return x.reshape(x.size(0), -1).sum(dim=-1)


def remove_mean_including_reactants_and_products(x):
    """
    Subtracts the mean of each row (excluding the One-Hot-Encoded atom types) from the corresponding row.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_atoms, features).

    Returns:
        torch.Tensor: Tensor with mean-subtracted rows.

    """

    mean_reactant = torch.mean(x[:, :, 4:7], dim=1, keepdim=True)
    mean_product = torch.mean(x[:, :, 7:10], dim=1, keepdim=True)
    mean_ts = torch.mean(x[:, :, 10:13], dim=1, keepdim=True)

    x[:, :, 4:7] = x[:, :, 4:7] - mean_reactant
    x[:, :, 7:10] = x[:, :, 7:10] - mean_product
    x[:, :, 10:13] = x[:, :, 10:13] - mean_ts

    return x

def remove_mean_just_ts(x):
    assert x.shape[2] == 3
    mean_ts = torch.mean(x, dim=1, keepdim=True)

    x = x - mean_ts

    return x


def assert_mean_zero(x):
    """
    Asserts that the mean of each row in the input tensor `x` (excluding the One-Hot-Encoded atom types) is approximately zero.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, features).

    Raises:
        AssertionError: If the absolute value of the maximum mean is greater than or equal to 1e-6.
    """
    mean = torch.mean(x, dim=1, keepdim=True)
    # print(mean.abs().max().item())
    # print(mean.shape)
    mean = mean.abs().max().item()
    # print(mean)
    assert mean < 1e-5

def remove_mean_with_mask(x, node_mask):
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x

def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.to(variable.device))).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'


def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask):
    assert len(size) == 3
    x = torch.randn(size, device=device)


    if len(node_mask.size()) == 2:  # Expand so that you can remove it from X
        node_mask = node_mask.unsqueeze(2).expand(size)

    x_masked = x * node_mask.to(device)

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean_with_mask(x_masked, node_mask.to(device))
    return x_projected


def sample_center_gravity_zero_gaussian(size, device):
    """
    Samples data from a zero-mean Gaussian distribution centered at the origin and projects it onto the space of transition states (TS).

    Args:
        size (tuple): Desired size of the sampled data in the format (batch_size, num_points, dimension).
        device (str or torch.device): Device to run the sampling on (e.g., "cpu", "cuda").

    Returns:
        torch.Tensor: Sampled data with the same shape as the input size, but centered around the origin and projected onto the TS space.

    Raises:
        AssertionError: If the length of the size tuple is not 3.
        AssertionError: If the third dimension of the size tuple is not 3, indicating that noise is only added to the TS.

    """
    assert len(size) == 3
    assert size[2] == 3  # Check that we are only adding noise to TS

    x = torch.randn(size, device=device)

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean_just_ts(x)
    return x_projected





def standard_gaussian_log_likelihood(x):
    """
    Computes the log-likelihood of a standard Gaussian distribution for a given input tensor.

    No Euclidian Distances are accounted for here unlike the centred-gravity function above. 

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Log-likelihood of the standard Gaussian distribution.

    """
    assert len(x.size()) == 3  # Check 3D
    assert x.shape[2] == 3  # Assert that we are only doing this for the TS
    log_px = sum_except_batch(-0.5 * x * x - 0.5 * np.log(2*np.pi))
    return log_px


def sample_gaussian(size, device):
    """
    Standard Gaussian Sampling - Whithout the removal of the centre of gravity --> NOT E(3) Equivariant.
    """
    assert len(size) == 3
    assert size[2] == 3  # Check that we are only adding noise to TS


    x = torch.randn(size, device=device)
    return x



# Rotation data augmntation
def random_rotation(x, h):
    """
    
    Adapted from: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/utils.py


    Instead of Only taking X, it should also perform the rotations on the Reactant and Prodcut Coordinates


    Perform Random Rotations during training to augment the model and train it on the rotation side of E(3)-Equivariance
    
    """
    bs, _, n_dims = x.size()
    device = x.device
    angle_range = np.pi * 2


    # Seperate the Reactant/Product from h: 
    product = h[:, :, -3:]
    reactant = h[:, :, -6:-3]



    assert n_dims == 3  # Make sure we have 3D Coordinates

    # Build Rx
    Rx = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
    theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi   # Random theta angle to rotate
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    Rx[:, 1:2, 1:2] = cos
    Rx[:, 1:2, 2:3] = sin
    Rx[:, 2:3, 1:2] = - sin
    Rx[:, 2:3, 2:3] = cos

    # Build Ry
    Ry = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
    theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    Ry[:, 0:1, 0:1] = cos
    Ry[:, 0:1, 2:3] = -sin
    Ry[:, 2:3, 0:1] = sin
    Ry[:, 2:3, 2:3] = cos

    # Build Rz
    Rz = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).to(device)
    theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    Rz[:, 0:1, 0:1] = cos
    Rz[:, 0:1, 1:2] = sin
    Rz[:, 1:2, 0:1] = -sin
    Rz[:, 1:2, 1:2] = cos


    # Perform the rotation on the Transition state
    x = x.transpose(1, 2)
    # Perform the rotation in X direction
    x = torch.matmul(Rx, x)
    # Perform the rotation in X direction
    x = torch.matmul(Ry, x)
    # Perform the rotation in X direction
    x = torch.matmul(Rz, x)
    # Transpose back the to the original shape
    x = x.transpose(1, 2)

    # Perform the rotation on the Reactant state
    reactant = reactant.transpose(1, 2)
    # Perform the rotation in X direction
    reactant = torch.matmul(Rx, reactant)
    # Perform the rotation in X direction
    reactant = torch.matmul(Ry, reactant)
    # Perform the rotation in X direction
    reactant = torch.matmul(Rz, reactant)
    # Transpose back the to the original shape
    reactant = reactant.transpose(1, 2)

    # Perform the rotation on the Product state
    product = product.transpose(1, 2)
    # Perform the rotation in X direction
    product = torch.matmul(Rx, product)
    # Perform the rotation in X direction
    product = torch.matmul(Ry, product)
    # Perform the rotation in X direction
    product = torch.matmul(Rz, product)
    # Transpose back the to the original shape
    product = product.transpose(1, 2)


    # Concatenate the reactant and product back together
    h[:, :, -3:] = product
    h[:, :, -6:-3] = reactant



    return x.contiguous(), h  # Returns contigous memory position





if __name__ == "__main__":
    print("Running Tests")

    # Let's check that all the above functions work as intended here !!!    
    dataset = QM90_TS(directory="Dataset/data/Clean_Geometries/")
    train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)


    example_sample = next(iter(train_loader))

    # print(next(iter(train_loader)))

# Now we can verify that each function works as intended: 
    
    # Sum_except_batch: 
    sum_batch = sum_except_batch(example_sample)
    # print(sum_batch)


    # Remove Mean:
    fake_batch_mean_removed = remove_mean_including_reactants_and_products(example_sample)
    # print((fake_batch_mean_removed == example_sample).sum())
    assert_mean_zero(fake_batch_mean_removed[:, :, 4:7])


    # Remove mean of TS: 
    TS_removed_mean = remove_mean_just_ts(example_sample[:, :, 10:13])
    # print(TS_removed_mean)


    # Assert mean is 0 --> it should already be the case: 
    assert_mean_zero(example_sample[:, : , 4:7])

    
    # Sample random gaussian like the shape of TS: 
    random_ts_noise = sample_center_gravity_zero_gaussian(example_sample[:, :, 10:].shape, "cpu")
    # print(random_ts_noise)
    assert_mean_zero(random_ts_noise)


    # Testing the other Log likelihood function: 
    log_hood_2 = standard_gaussian_log_likelihood(example_sample[:, :, 10:])
    # print(log_hood_2)


    # Testing the Rajdom gaussian sampling that is not centred on 0: 
    random_noise = sample_gaussian(example_sample[:, :, 10:13].shape, "cpu")
    # assert_mean_zero(random_noise)  # Should cause an error

