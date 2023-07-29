"""

Script for Equivariant diffusion model:
---------------------------------------

Code adapted from https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/equivariant_diffusion/en_diffusion.py

Adaptations: 
1. Cleaned whole script
    1.1 Removed unused functions and asserts
    1.2 Removed Variational-Lower-Bound Loss as not used
    1.3 More
    1.4 Seperated functions for modularity
2. Changed so Noising on H is not re-integrated into the model - as we are only trying to predict X part of xH
"""

import os
import numpy as np
import torch
from torch.nn import functional as F
import Diffusion.utils as diffusion_utils
from tqdm import tqdm
from torch.nn import MSELoss
import wandb
from sklearn.model_selection import train_test_split

from Dataset_W93.dataset_class import * 
from src.EGNN import model_dynamics_with_mask
from Diffusion.noising import PredefinedNoiseSchedule





# Defining some useful util functions.
def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)




class DiffusionModel(torch.nn.Module):
    """
    The E(n) Diffusion Module.
    """
    def __init__(
            self,
            dynamics: model_dynamics_with_mask.EGNN_dynamics_QM9, in_node_nf: int, n_dims: int, device="cpu",
            timesteps: int = 1000, noise_schedule='cosine',
            noise_precision=1e-4, norm_values=(1., 1., 1.),
            norm_biases=(None, 0., 0.)):
        super().__init__()


        # Setup GPU:
        self.device = device
        
        # MSE is used as the L_simple from paper: 
        self.mse = MSELoss()


        # Setup the noise schedule and get Gamma Values: 
        self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps,
                                                 precision=noise_precision)

        # Denoising EGNN Network:
        self.dynamics = dynamics

        # Setup EGNN variables:
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.num_classes = self.in_node_nf

        # Number of Sampling Timesteps:
        self.T = timesteps


        # These variables are a bit Random/Useless in our case
        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer('buffer', torch.zeros(1))

        # Check that the noise schedule allows for the total noising process to reach Gaussian Noise:
        self.check_issues_norm_values()

    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        max_norm_value = max(self.norm_values[1], self.norm_values[2])

        if sigma_0 * num_stdevs > 1. / max_norm_value:
            raise ValueError(
                f'Value for normalization value {max_norm_value} probably too '
                f'large with sigma_0 {sigma_0:.5f} and '
                f'1 / norm_value = {1. / max_norm_value}')

    def phi(self, x, t, node_mask, edge_mask, context):
        _, net_out = self.dynamics._forward(t, x, node_mask, edge_mask, context = context)  # Predicted H is USELESS IN our use-case
        # net_out = diffusion_utils.remove_mean_with_mask(net_out,  node_mask.unsqueeze(2).expand(net_out.size()))
        diffusion_utils.assert_mean_zero_with_mask(net_out, node_mask.unsqueeze(2).expand(net_out.size()))
        return net_out

    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)

    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(1), dim=1)  # Changed from squeeze(2)
        return (number_of_nodes - 1) * self.n_dims


    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -expm1(softplus(gamma_s) - softplus(gamma_t)), target_tensor
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(
            alpha_t_given_s, target_tensor)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s



    def compute_x_pred(self, net_out, zt, gamma_t, final = False):
        """Commputes x_pred, i.e. the most likely prediction of x."""
        sigma_t = self.sigma(gamma_t, target_tensor=net_out)
        alpha_t = self.alpha(gamma_t, target_tensor=net_out)
        eps_t = net_out
        x_pred = 1. / alpha_t * (zt - sigma_t * eps_t)  # This is from the prediction formula fron noise to prediction value
        return x_pred

    def compute_error(self, net_out, gamma_t, eps):
        """Computes error, i.e. the most likely prediction of x."""
        error = self.mse(eps, net_out)
        return error




    def log_constants_p_x_given_z0(self, x, node_mask):
        """Computes p(x|z0)."""
        batch_size = x.size(0)

        n_nodes = node_mask.squeeze(1).sum(1)  # N has shape [B]  # Changed it from squeeze 2
        assert n_nodes.size() == (batch_size,)
        degrees_of_freedom_x = (n_nodes - 1) * self.n_dims

        zeros = torch.zeros((x.size(0), 1), device=x.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x.to(self.device) * (- log_sigma_x.to(self.device) - 0.5 * np.log(2 * np.pi))




    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context=None, fix_noise=False):
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)

        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)

        net_out = self.phi(t=zeros, x=z0, node_mask=node_mask, edge_mask=edge_mask, context=context)        
        
        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out= net_out.to(self.device), zt = z0[:, :, -3:].to(self.device), gamma_t=gamma_0.to(self.device))

        # xh = self.sample_normal(mu=mu_x.to(self.device), sigma=sigma_x.to(self.device), node_mask=node_mask.to(self.device), fix_noise=fix_noise)
        x = self.sample_normal(mu=mu_x.to(self.device), sigma=sigma_x.to(self.device), node_mask=node_mask.to(self.device), fix_noise=fix_noise)

        return x

    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_position(bs, mu.size(1), node_mask)
        return mu + sigma * eps
    

    def sample_position(self, n_samples, n_nodes, node_mask):  # Was previously sample_combined_position_feature_noise
        """
        Samples mean-centered normal noise for z_x.
        """
        z_x = diffusion_utils.sample_center_gravity_zero_gaussian_with_mask(size=(n_samples, n_nodes, self.n_dims), device=self.device, node_mask=node_mask)
        diffusion_utils.assert_mean_zero_with_mask(z_x, node_mask.unsqueeze(2).expand(z_x.size()))
        return z_x


    def compute_loss(self, x, h, node_mask, edge_mask, context, t0_always):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE)."""
        
        # This part is about whether to include loss term 0 always.
        if t0_always:
            # loss_term_0 will be computed separately.
            # estimator = loss_0 + loss_t,  where t ~ U({1, ..., T})
            lowest_t = 1
        else:
            # estimator = loss_t,           where t ~ U({0, ..., T})
            lowest_t = 0

        # Sample a timestep t.
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(x.size(0), 1), device=x.device).float()
        # s_int = t_int - 1

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        # s = s_int / self.T
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        # gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = self.sample_position(n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)
  
        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        # z_t = alpha_t.to(self.device) * xh.to(self.device) + sigma_t.to(self.device) * eps.to(self.device)
        z_t = alpha_t.to(self.device) * x.to(self.device) + sigma_t.to(self.device) * eps.to(self.device)


        diffusion_utils.assert_mean_zero_with_mask(z_t, node_mask.unsqueeze(2).expand(z_t.size()).to(self.device))
        
        # Now Concatenate x with H before feeding it into the model:  added by me
        z_t = torch.cat([h.to(self.device), z_t.to(self.device)], dim=2)


        # Neural net prediction.
        net_out = self.phi(z_t.to(self.device), t.to(self.device), node_mask.to(self.device), edge_mask.to(self.device), context)

        # Compute the error.
        error = self.compute_error(net_out, gamma_t, eps)

        return error

    def forward(self, x, h, node_mask=None, edge_mask=None, context=None):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        """
        # Normalize data, take into account volume change in x.
        #x  , h, delta_log_px = self.normalize(x, h, node_mask)

        loss = self.compute_loss(x, h, node_mask, edge_mask, context=None, t0_always=False)

        neg_log_pxh = loss

        # Correct for normalization on x.
        # assert neg_log_pxh.size() == delta_log_px.size()
        # print(delta_log_px)
        neg_log_pxh = neg_log_pxh.to(self.device) #  - delta_log_px.mean().to(self.device)

        return neg_log_pxh

    def sample_p_zs_given_zt(self, s, t, zt, node_mask, edge_mask, context=None, fix_noise=False):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)


        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        eps_t = self.phi(zt, t, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        diffusion_utils.assert_mean_zero_with_mask(zt[:, :, -3:], node_mask.unsqueeze(2).expand(zt[:, :, -3:].size()))
        diffusion_utils.assert_mean_zero_with_mask(eps_t[:, :, -3:], node_mask.unsqueeze(2).expand(eps_t[:, :, -3:].size()))

        # mu = zt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
        mu = zt[:, :, -3:].to(self.device) / alpha_t_given_s.to(self.device) - (sigma2_t_given_s.to(self.device) / alpha_t_given_s.to(self.device) / sigma_t.to(self.device)) * eps_t.to(self.device)




        # Compute sigma for p(zs | zt). 
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu.to(self.device), sigma.to(self.device), node_mask.to(self.device), fix_noise)

        # Project down to avoid numerical runaway of the center of gravity.
        # Have to concatenate the OLD H: 
        # first remove mean: 
        zs = torch.cat([zt[:, :, :-3].to(self.device), diffusion_utils.remove_mean_with_mask(zs[:, :, -3:], node_mask.unsqueeze(2).expand(zs[:,:,-3:].size())).to(self.device)], dim=2)
        return zs


    @torch.no_grad()
    def sample(self, h, n_samples, n_nodes, node_mask, edge_mask, context=None, context_size=0, fix_noise=False):
        """
        Draw samples from the generative model.
        """

        # Predefined_h compared to theirs
        predefined_h = h

        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_position(1, n_nodes, node_mask)
        else:
            z = self.sample_position(n_samples, n_nodes, node_mask)


        # Concatenate the predefined H: 
        z = torch.cat([predefined_h.to(self.device), z.to(self.device)], dim=2)

        # Check the coordinates have zero mean
        diffusion_utils.assert_mean_zero_with_mask(z[:, :, -3:], node_mask.unsqueeze(2).expand(z[:, :, -3:].size()))
        diffusion_utils.assert_mean_zero_with_mask(z[:, :, 7+context_size:10+context_size], node_mask.unsqueeze(2).expand(z[:, :,  7+context_size:10+context_size].size()))
        diffusion_utils.assert_mean_zero_with_mask(z[:, :, 4+context_size:7+context_size], node_mask.unsqueeze(2).expand(z[:, :, 4+context_size:7+context_size].size()))


        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(s_array.to(self.device), t_array.to(self.device), z.to(self.device), node_mask.to(self.device), edge_mask.to(self.device), context, fix_noise=fix_noise)

        # Finally sample p(x, h | z_0).
        x = self.sample_p_xh_given_z0(z, node_mask.to(self.device), edge_mask.to(self.device), context, fix_noise=fix_noise)

        diffusion_utils.assert_mean_zero_with_mask(x, node_mask.unsqueeze(2).expand(x.size()))

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                  f'the positions down.')
            x = diffusion_utils.remove_mean_with_mask(x, node_mask.unsqueeze(2).expand(x.size()))
        return x


    @torch.no_grad()
    def sample_chain(self, h, n_samples, n_nodes, node_mask, edge_mask, keep_frames=None, context=None, context_size=0, fix_noise=False):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """

        predefined_h = h.to(self.device)

        z = self.sample_position(n_samples, n_nodes, node_mask)

        diffusion_utils.assert_mean_zero(z[:, :, -3:])  # Might have to include mask

        # Concatenate predefined H:
        z = torch.cat([predefined_h, z], dim=2)
        


        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(s_array.to(self.device), 
                                          t_array.to(self.device), 
                                          z.to(self.device), 
                                          node_mask.to(self.device), 
                                          edge_mask.to(self.device))


            diffusion_utils.assert_mean_zero(z[:, :, -3:])

            # Write to chain tensor
            write_index = (s * keep_frames) // self.T
            chain[write_index] = z

        # Finally sample p(x, h | z_0).
        x = self.sample_p_xh_given_z0(z,
                                      node_mask.to(self.device), 
                                      edge_mask.to(self.device))

        diffusion_utils.assert_mean_zero(x[:, :, -3:])

        xh = torch.cat([h, x], dim=2)
        chain[0] = xh  # Overwrite last frame with the resulting x and h.

        chain_flat = chain.view(n_samples * keep_frames, *z.size()[1:])

        return chain_flat

if __name__ == "__main__":
    print("running script")


    # Setup the device: 
    device = model_dynamics_with_mask.setup_device()


    remove_hydrogens = False
    include_context = True
            
    if remove_hydrogens:
        in_node_nf = 9 + 1  # To account for time and 1 less OHE
    else:
        in_node_nf = 10 + 1  # To account for time
    
    if include_context:
        in_node_nf += 1  

    out_node = 3
    context_nf = 0 
    n_dims = 3
    noise_schedule = "cosine"
    timesteps = 2_000
    batch_size = 64
    n_layers = 3
    hidden_features = 64
    lr = 8e-4
    epochs = 100


    # Setup for clear model Tracking:
    model_name = f"{n_layers}_layers_{hidden_features}_hiddenfeatures_{lr}_lr_{noise_schedule}_{timesteps}_timesteps_{batch_size}_batch_size_{epochs}_epochs_{remove_hydrogens}_Rem_Hydrogens"
    folder_name = "Diffusion/Clean/" + model_name + "/"
    
    # Setup WandB:
    wandb.init(project='Diffusion_context_test', name=model_name)
    wandb.config.name = model_name
    wandb.config.batch_size = batch_size
    wandb.config.epochs = epochs
    wandb.config.lr = lr
    wandb.config.hidden_node_features = hidden_features
    wandb.config.number_of_layers= n_layers



    denoising_model = model_dynamics_with_mask.EGNN_dynamics_QM9(in_node_nf=in_node_nf, context_node_nf=context_nf, hidden_nf=hidden_features, out_node=out_node, n_dims=n_dims, sin_embedding=True, n_layers=n_layers, device=device)  # Something is going wrong with the embedding of H


    dataset = QM90_TS(directory="Dataset_W93/data/Clean_Geometries/", remove_hydrogens=remove_hydrogens, include_context=include_context)

    # Calculate the sizes for each split
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.3, random_state=42)

    # Splitting the train dataset into train and validation sets
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1, random_state=42)

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)  # Not shuffled so that we can visualise the same samples 



    # Setup the diffusion model: 
    diffusion_model = DiffusionModel(dynamics= denoising_model, in_node_nf=in_node_nf, n_dims=n_dims, timesteps=timesteps, device=device, noise_schedule=noise_schedule)


    # Now let's try training the function and see if it works for now
    optimiser = torch.optim.Adam(diffusion_model.parameters(), lr = lr)

    diffusion_model.to(device)
    for epoch in range(epochs):
            total_train_loss = 0.
            total_val_loss = 0.

            # Setup training mode: 
            diffusion_model.train()
            for batch, node_mask in tqdm(train_loader):


                # Check that the values are centred: 
                diffusion_utils.assert_mean_zero_with_mask(batch[:, :, -3:], node_mask.unsqueeze(2).expand(batch[:, :, -3:].size()))
                
                optimiser.zero_grad()


                h = batch[:, :, :-3].to(device)
                x = batch[:, :, -3:].to(device)

                # setup the edge_mask: 
                edge_mask = node_mask.unsqueeze(1)  * node_mask.unsqueeze(2)

                # Create mask for diagonal, as atoms cannot connect to themselves: 
                diag_mask = ~torch.eye(edge_mask.size(-1), device=edge_mask.device).unsqueeze(0).bool()

                # Expand to batch size: 
                diag_mask = diag_mask.expand(edge_mask.size())

                # Multiply the edge mask by the diagonal mask to not have connections with itself: 
                edge_mask *= diag_mask       



                # Calculate the loss: 
                nll = diffusion_model(x.to(device), h.to(device), node_mask.to(device), edge_mask.to(device))

                loss = nll.to(device)

                loss.backward()
                optimiser.step()

                total_train_loss += nll
            
            total_train_loss /= len(train_loader)
            print(f"At epoch {epoch} \t Train Loss = {total_train_loss}")
            wandb.log({"Train_loss": total_train_loss})

            # Setup Validation part: 
            diffusion_model.eval()
            with torch.no_grad():
                for batch, node_mask in tqdm(val_loader):


                    h = batch[:, :, :-3].to(device)
                    x = batch[:, :, -3:].to(device)


                    # setup the edge_mask: 
                    edge_mask = node_mask.unsqueeze(1)  * node_mask.unsqueeze(2)

                    # Create mask for diagonal, as atoms cannot connect to themselves: 
                    diag_mask = ~torch.eye(edge_mask.size(-1), device=edge_mask.device).unsqueeze(0).bool()

                    # Expand to batch size: 
                    diag_mask = diag_mask.expand(edge_mask.size())

                    # Multiply the edge mask by the diagonal mask to not have connections with itself: 
                    edge_mask *= diag_mask


                    # Calculate the loss: 
                    nll = diffusion_model(x.to(device), h.to(device), node_mask.to(device), edge_mask.to(device))
                    loss = nll.to(device)
                    total_val_loss += nll
                
            total_val_loss /= len(val_loader)
            print(f"At epoch {epoch} \t Val Loss = {total_val_loss}")
            wandb.log({"val_loss": total_val_loss})


    # Test the whole test set: 
    total_test_loss = 0
    diffusion_model.eval()
    with torch.no_grad():
        for batch, node_mask in tqdm(test_loader):


            h = batch[:, :, :-3].to(device)
            x = batch[:, :, -3:].to(device)


            # setup the edge_mask: 
            edge_mask = node_mask.unsqueeze(1)  * node_mask.unsqueeze(2)

            # Create mask for diagonal, as atoms cannot connect to themselves: 
            diag_mask = ~torch.eye(edge_mask.size(-1), device=edge_mask.device).unsqueeze(0).bool()

            # Expand to batch size: 
            diag_mask = diag_mask.expand(edge_mask.size())

            # Multiply the edge mask by the diagonal mask to not have connections with itself: 
            edge_mask *= diag_mask


            # Calculate the loss: 
            nll = diffusion_model(x.to(device), h.to(device), node_mask.to(device), edge_mask.to(device))
            loss = nll.to(device)
            total_test_loss += nll
        
    total_test_loss /= len(val_loader)
    print(f"Total MSE Test Loss is:\t{total_test_loss}\n")


    # Save the model: 
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    model_path = folder_name + f"Weights_Test_MSE_{total_test_loss:.3f}/"
    sample_path = folder_name + "Samples/"
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    # Set the file path for saving the model weights
    model_path = os.path.join(model_path, f"weights.pt")

    # Save model
    torch.save(diffusion_model.state_dict(), model_path)




