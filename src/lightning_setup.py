import os
import pytorch_lightning
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import wandb


from src.Diffusion.equivariant_diffusion import (
    DiffusionModel,
    DiffusionModel_graph,
    get_node_features,
)


from src.Diffusion.saving_sampling_functions import write_xyz_file, return_xyz
from src.Diffusion.utils import random_rotation
from src.EGNN import dynamics
from src.EGNN.dynamics_with_graph import (
    EGNN_dynamics_graph,
)
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau


from data.Dataset_W93.dataset_class import W93_TS
from data.Dataset_W93.dataset_reactant_and_product_graph import (
    W93_TS_coords_and_reacion_graph,
)
from data.Dataset_TX1.dataset_TX1_class import TX1_dataset
from data.Dataset_RGD1.RGD1_dataset_class import RGD1_TS

from pytorch_lightning.callbacks import LearningRateMonitor


"""
Script to setup diffusion model using python Lightning
------------------------------------------------------
"""


# Pytorch Lightning class for the diffusion model:
class LitDiffusionModel(pl.LightningModule):
    def __init__(
        self,
        dataset_to_use,
        in_node_nf,
        hidden_features,
        n_layers,
        device,
        lr,
        remove_hydrogens,
        test_sampling_number,
        save_samples,
        save_path,
        timesteps,
        noise_schedule,
        random_rotations=False,
        augment_train_set=False,
        include_context=False,
        learning_rate_schedule=False,
        no_product=False,
        batch_size=64,
        pytest_time=False,
    ):
        super(LitDiffusionModel, self).__init__()
        self.dataset_to_use = dataset_to_use
        self.pytest_time = pytest_time
        self.batch_size = batch_size
        self.lr = lr
        self.remove_hydrogens = remove_hydrogens
        self.include_context = include_context
        self.learning_rate_schedule = learning_rate_schedule
        self.no_product = no_product

        if save_samples:
            assert (
                save_path is not None
            ), "Make sure that a Path to where you want the samples to be placed is given please"  # noqa

        if self.no_product:
            assert (
                not augment_train_set
            ), "Cannot augment the train set if the products are not used"

        if self.pytest_time:
            assert (
                self.dataset_to_use == "W93"
            ), "Pytest can only be used with the W93 dataset"

        # Assert you have the correct datasets:
        assert (
            self.dataset_to_use == "W93"
            or self.dataset_to_use == "TX1"
            or self.dataset_to_use == "RGD1"
        ), "Dataset can only be W93 for TS-DIFF dataset or TX1 for the MIT paper OR the new RGD1 dataset"  # noqa

        # Add asserts for what is possible with the TX1 dataset and what is possible with the W93 dataset  # noqa
        if self.dataset_to_use == "TX1":
            assert (
                not self.include_context
            ), "For the TX1 dataset, including context is not allowed."
            assert (
                not self.remove_hydrogens
            ), "For the TX1 dataset, removing hydrogens is not allowed."

        # Add asserts for what is possible with the TX1 dataset and what is possible with the W93 dataset  # noqa
        if self.dataset_to_use == "RGD1":
            assert (
                not self.include_context
            ), "For the RGD1 dataset, including context is not allowed."
            # assert (
            #     not self.remove_hydrogens
            # ), "For the RGD1 dataset, removing hydrogens is not allowed."

        # For now, context is only 1 variable added to the node features - Only possible with the W93 dataset - Try and not make this hard-coded  # noqa
        if self.include_context and self.dataset_to_use == "W93":
            self.context_size = 1
        else:
            self.context_size = 0

        # Variables for testing (Number of samples and if we want to save the samples):  # noqa
        self.test_sampling_number = test_sampling_number
        self.save_samples = save_samples

        # Include Random Rotations of the molecules so that it is different at each epoch:  # noqa
        self.data_random_rotations = random_rotations

        # Augment train set by duplicating train reactions but reversing reactants and products with each-other:  # noqa
        self.augment_train_set = augment_train_set

        if self.save_samples:
            # assert that the save_path input is correct:
            assert os.path.exists(save_path)
            self.save_path = save_path

        if self.dataset_to_use == "W93":
            # At testing time only test with sub-folder
            if self.pytest_time:
                self.dataset = W93_TS(
                    directory="data/Dataset_W93/example_data_for_testing/Clean_Geometries",  # noqa
                    remove_hydrogens=self.remove_hydrogens,
                    include_context=include_context,
                )

            else:
                self.dataset = W93_TS(
                    directory="data/Dataset_W93/data/Clean_Geometries",
                    remove_hydrogens=self.remove_hydrogens,
                    include_context=include_context,
                )

            # # Split into 8:1:1 ratio:
            self.train_dataset, test_dataset = train_test_split(
                self.dataset, test_size=0.2, random_state=42
            )
            self.val_dataset, self.test_dataset = train_test_split(
                test_dataset, test_size=0.5, random_state=42
            )

        elif self.dataset_to_use == "TX1":
            self.dataset = TX1_dataset(split="data")

            # We want to match the sizes used in MIT paper 9,000 for training and 1,073 for testing  # noqa
            train_dataset, self.test_dataset = train_test_split(
                self.dataset, test_size=0.1, random_state=42, shuffle=True
            )

            # Split the training set into a 8:1 split to train/val set:
            self.train_dataset, self.val_dataset = train_test_split(
                train_dataset, test_size=1 / 9, random_state=42, shuffle=True
            )

        elif self.dataset_to_use == "RGD1":
            self.dataset = RGD1_TS(
                directory="data/Dataset_RGD1/data/Single_and_Multiple_TS",
                remove_hydrogens=self.remove_hydrogens,
            )  # noqa

            # We want to match the sizes used in MIT paper 9,000 for training and 1,073 for testing  # noqa
            train_dataset, self.test_dataset = train_test_split(
                self.dataset, test_size=0.1, random_state=42, shuffle=True
            )

            # Split the training set into a 8:1 split to train/val set:
            self.train_dataset, self.val_dataset = train_test_split(
                train_dataset, test_size=1 / 9, random_state=42, shuffle=True
            )

        # Augment the train set here:
        if self.augment_train_set:
            self.augment_reactants_products()

        # Setup the denoising model:
        self.denoising_model = dynamics.EGNN_dynamics_QM9(
            in_node_nf=in_node_nf,
            hidden_nf=hidden_features,
            sin_embedding=True,
            n_layers=n_layers,
            device=device,
            attention=False,
        )

        # Setup the diffusion model:
        self.diffusion_model = DiffusionModel(
            dynamics=self.denoising_model,
            in_node_nf=in_node_nf,
            timesteps=timesteps,
            noise_schedule=noise_schedule,
            device=device,
        )

        # Save the Hyper-params used:
        self.save_hyperparameters()

    def augment_reactants_products(self):
        """
        This function will augment the train set
        """
        print(
            "Augmenting the train set, by duplicating train reactions and replacing reactants with products and vice-versa"  # noqa
        )
        train_set_augmented = []

        for data in self.train_dataset:
            train_set_augmented.append(data)  # Append the original data
            # Seperate into graphs and node masks
            molecule, node_mask = data[0], data[1]

            if self.remove_hydrogens:
                # Swap the reactant and product - but keep the OHE
                swapped_molecule = torch.cat(
                    (
                        molecule[:, : 3 + self.context_size],
                        molecule[
                            :, 6 + self.context_size : 9 + self.context_size  # noqa
                        ],  # noqa
                        molecule[
                            :, 3 + self.context_size : 6 + self.context_size  # noqa
                        ],  # noqa
                        molecule[:, 9 + self.context_size :],  # noqa
                    ),
                    dim=1,
                )

            else:
                swapped_molecule = torch.cat(
                    (
                        molecule[:, : 4 + self.context_size],
                        molecule[
                            :, 7 + self.context_size : 10 + self.context_size  # noqa
                        ],  # noqa
                        molecule[
                            :, 4 + self.context_size : 7 + self.context_size  # noqa
                        ],  # noqa
                        molecule[:, 10 + self.context_size :],  # noqa
                    ),
                    dim=1,
                )

            # Now we can create the tuple with the node mask (No Need to swap anything within the node_mask)  # noqa
            swapped_data = (swapped_molecule, node_mask)
            # Append it to the train set:
            train_set_augmented.append(swapped_data)

        # Swap the train set with the augmented one:
        self.train_dataset = train_set_augmented

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=24,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=24,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=24,
        )  # No need for shuffling - Will make visualisation easier.

    def configure_optimizers(self):
        """
        Setup Optimiser and learning rate scheduler if needed.
        """
        optimizer = torch.optim.Adam(
            self.diffusion_model.parameters(), lr=self.lr
        )  # noqa
        if self.learning_rate_schedule:
            lr_scheduler = {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.5,
                    patience=50,
                    min_lr=self.lr / 100,  # noqa
                ),
                "monitor": "val_loss",  # The metric to monitor
                "interval": "epoch",  # The interval to invoke the scheduler ('epoch' or 'step')  # noqa
                "frequency": 1,  # The frequency of scheduler invocation (every 1 epoch in this case)  # noqa
            }
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def forward(self, x, h, node_mask, edge_mask):
        return self.diffusion_model(x, h, node_mask, edge_mask)

    def training_step(self, batch, batch_idx):
        # Get Coordinates and node mask:
        coords, node_mask = batch

        # Split the coords into H and X vectors:
        if self.no_product:
            h = coords[:, :, :-6]
        else:
            h = coords[:, :, :-3]
        x = coords[:, :, -3:]

        # Setup the Edge mask (1 everywhere except the diagonals - atom cannot be connected to itself):  # noqa
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = (
            ~torch.eye(edge_mask.size(-1), device=edge_mask.device)
            .unsqueeze(0)
            .bool()  # noqa
        )
        diag_mask = diag_mask.expand(edge_mask.size())
        edge_mask *= diag_mask

        # Use random rotations during training if required:
        if self.data_random_rotations:
            x, h = random_rotation(x, h)

        # Forward pass:
        loss = self(x, h, node_mask, edge_mask)

        # Log the loss:
        self.log(
            "train_loss", loss, on_epoch=True, prog_bar=True, on_step=False
        )  # noqa
        return loss

    def validation_step(self, batch, batch_idx):
        # Get Coordinates and node mask:
        coords, node_mask = batch

        # Split the coords into H and X:
        if self.no_product:
            h = coords[:, :, :-6]
        else:
            h = coords[:, :, :-3]
        x = coords[:, :, -3:]

        # Setup the Edge mask (1 everywhere except the diagonals - atom cannot be connected to itself):  # noqa
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = (
            ~torch.eye(edge_mask.size(-1), device=edge_mask.device)
            .unsqueeze(0)
            .bool()  # noqa
        )
        diag_mask = diag_mask.expand(edge_mask.size())
        edge_mask *= diag_mask

        # Forward pass:
        loss = self(x, h, node_mask, edge_mask)

        # Log the loss:
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def sample_and_test(
        self,
        true_h,
        true_x,
        node_mask,
        edge_mask,
        folder_path,
        remove_hydrogens=False,
        device=None,
    ):
        number_samples = self.test_sampling_number
        # Get the true reactant and product - Still hard-coded:
        if remove_hydrogens:
            true_reactant = true_h[
                :, :, 3 + self.context_size : 6 + self.context_size  # noqa
            ].clone()
            if not self.no_product:
                true_product = true_h[
                    :, :, 6 + self.context_size : 9 + self.context_size  # noqa
                ].clone()

            # Get the OHE of atom-type:
            atom_ohe = true_h[:, :, :3]

        else:
            true_reactant = true_h[
                :, :, 4 + self.context_size : 7 + self.context_size  # noqa
            ].clone()
            if not self.no_product:
                true_product = true_h[
                    :, :, 7 + self.context_size : 10 + self.context_size  # noqa
                ].clone()

            # Get the OHE of atom-type:
            atom_ohe = true_h[:, :, :4]

        # Inflate H so that it is the size of the number of samples:
        inflated_h = true_h.repeat(number_samples, 1, 1)

        # Inflate the node mask and edge masks:
        node_mask = node_mask.repeat(number_samples, 1)
        edge_mask = edge_mask.repeat(number_samples, 1, 1)

        # Set model to evaluation mode (Faster computations and no data-leakage):  # noqa
        self.diffusion_model.eval()
        if self.pytest_time:  # Different number of maximum number of atoms     # noqa
            if self.remove_hydrogens:
                samples = self.diffusion_model.sample(
                    inflated_h,
                    number_samples,
                    6,
                    node_mask.to(device),
                    edge_mask.to(device),
                    context_size=self.context_size,
                )  # still hard coded
            else:
                samples = self.diffusion_model.sample(
                    inflated_h,
                    number_samples,
                    16,
                    node_mask.to(device),
                    edge_mask.to(device),
                    context_size=self.context_size,
                )  # still hard coded
        elif remove_hydrogens:
            samples = self.diffusion_model.sample(
                inflated_h,
                number_samples,
                7,
                node_mask.to(device),
                edge_mask.to(device),
                context_size=self.context_size,
            )  # still hard coded
        elif (
            self.dataset_to_use == "RGD1"
        ):  # Different number of maximum number of atoms     # noqa
            samples = self.diffusion_model.sample(
                inflated_h,
                number_samples,
                33,
                node_mask.to(device),
                edge_mask.to(device),
                context_size=self.context_size,
            )  # still hard coded
        else:
            samples = self.diffusion_model.sample(
                inflated_h,
                number_samples,
                23,
                node_mask.to(device),
                edge_mask.to(device),
                context_size=self.context_size,
            )  # Still hard coded

        # Round to prevent downstream type issues in RDKit:
        true_x = torch.round(true_x, decimals=3)

        # Concatenate the atom ohe with the true sample, reactant and product:
        true_sample = torch.cat([atom_ohe.to(device), true_x.to(device)], dim=2)  # noqa
        true_reactant = torch.cat(
            [atom_ohe.to(device), true_reactant.to(device)], dim=2
        )
        if not self.no_product:
            true_product = torch.cat(
                [atom_ohe.to(device), true_product.to(device)],
                dim=2,
            )

        # Convert to XYZ format:
        true_samples = return_xyz(
            true_sample,
            ohe_dictionary=self.dataset.ohe_dict,
            remove_hydrogen=remove_hydrogens,
        )
        true_reactant = return_xyz(
            true_reactant,
            ohe_dictionary=self.dataset.ohe_dict,
            remove_hydrogen=remove_hydrogens,  # noqa
        )
        if not self.no_product:
            true_product = return_xyz(
                true_product,
                ohe_dictionary=self.dataset.ohe_dict,
                remove_hydrogen=remove_hydrogens,  # noqa
            )

        # Save the true reactants/products/TS if save_samples set to true:
        if self.save_samples:
            true_filename = os.path.join(folder_path, "true_sample.xyz")
            write_xyz_file(true_samples, true_filename)

            reactant_filename = os.path.join(folder_path, "true_reactant.xyz")
            write_xyz_file(true_reactant, reactant_filename)

            if not self.no_product:
                product_filename = os.path.join(
                    folder_path,
                    "true_product.xyz",
                )
                write_xyz_file(true_product, product_filename)

        for i in range(number_samples):
            predicted_sample = (
                samples[i].unsqueeze(0).to(torch.float64)
            )  # Unsqueeeze so it has bs

            # Need to round to make sure all the values are clipped and can be converted to doubles when using RDKit down the line:  # noqa
            predicted_sample = torch.round(predicted_sample, decimals=3)

            predicted_sample = torch.cat(
                [atom_ohe.to(device), predicted_sample], dim=2
            )  # noqa

            # Return it to xyz format:
            predicted_sample = return_xyz(
                predicted_sample,
                ohe_dictionary=self.dataset.ohe_dict,
                remove_hydrogen=remove_hydrogens,  # noqa
            )

            # If the save samples is True then save it
            if self.save_samples:
                # Let's now try and save the molecule before aligning it and after to see the overlaps later on  # noqa
                aft_aligning_path = os.path.join(folder_path, f"sample_{i}.xyz")  # noqa

                # Save the samples:
                write_xyz_file(predicted_sample, aft_aligning_path)

    def test_step(self, batch, batch_idx):
        # Sample a bunch of test samples and then
        test_coords, test_node_mask = batch

        # Setup the edge mask:
        test_edge_mask = test_node_mask.unsqueeze(1) * test_node_mask.unsqueeze(  # noqa
            2
        )  # noqa
        diag_mask = (
            ~torch.eye(test_edge_mask.size(-1), device=test_edge_mask.device)
            .unsqueeze(0)
            .bool()
        )
        diag_mask = diag_mask.expand(test_edge_mask.size())
        test_edge_mask *= diag_mask

        # If save_samples is true then make sure we have a folder for each samples:  # noqa
        if self.save_samples:
            self.save_path_batch = self.save_path + f"batch_{batch_idx}/"
            os.makedirs(self.save_path_batch, exist_ok=True)

        # Iterate over each test compound and create samples:
        for i in range(test_coords.shape[0]):
            # Create a subfolder for the molecule:
            if self.save_samples:
                self.save_path_mol = self.save_path_batch + f"mol_{i}/"
                os.makedirs(self.save_path_mol, exist_ok=True)
            else:
                self.save_path_mol = None

            # Seperate the Samples and then feed them through sampling method:
            # Split the coords into H and X:
            if self.no_product:
                test_h = test_coords[i, :, :-6].unsqueeze(0)
            else:
                test_h = test_coords[i, :, :-3].unsqueeze(0)
            # test_h = test_coords[i, :, :-3].unsqueeze(0)
            test_x = test_coords[i, :, -3:].unsqueeze(0)
            node_mask_input = test_node_mask[i].unsqueeze(0)
            edge_mask_input = test_edge_mask[i].unsqueeze(0)

            # Create samples:
            self.sample_and_test(
                test_h,
                test_x,
                node_mask_input,
                edge_mask_input,
                self.save_path_mol,
                self.remove_hydrogens,
                self.device,
            )


class LitDiffusionModel_With_graph(pl.LightningModule):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_features,
        n_layers,
        device,
        lr,
        test_sampling_number,
        save_samples,
        save_path,
        timesteps,
        noise_schedule,
        learning_rate_schedule=False,
        no_product=False,
        batch_size=64,
        pytest_time=False,
    ):
        super(LitDiffusionModel_With_graph, self).__init__()
        self.pytest_time = pytest_time
        self.batch_size = batch_size
        self.lr = lr
        self.learning_rate_schedule = learning_rate_schedule
        self.no_product = no_product

        # Variables for testing (Number of samples and if we want to save the samples):  # noqa
        self.test_sampling_number = test_sampling_number
        self.save_samples = save_samples

        if self.save_samples:
            # assert that the save_path input is correct:
            assert os.path.exists(save_path)
            self.save_path = save_path

        # Setup the dataset:
        if self.pytest_time:
            dir = "data/Dataset_W93/example_data_for_testing/Clean_Geometries"
            self.dataset = W93_TS_coords_and_reacion_graph(
                directory=dir,
                running_pytest=self.pytest_time,
            )
        else:
            self.dataset = W93_TS_coords_and_reacion_graph()
        # Split into 8:1:1 ratio:
        self.train_dataset, test_dataset = train_test_split(
            self.dataset, test_size=0.2, random_state=42
        )
        self.val_dataset, self.test_dataset = train_test_split(
            test_dataset, test_size=0.5, random_state=42
        )

        # Setup the denoising model:
        self.denoising_model = EGNN_dynamics_graph(
            in_node_nf=in_node_nf,
            hidden_nf=hidden_features,
            in_edge_nf=in_edge_nf,
            out_node=3,
            n_dims=3,
            sin_embedding=True,
            n_layers=n_layers,
            device=device,
            attention=False,
        )
        # Setup the diffusion model:
        self.diffusion_model = DiffusionModel_graph(
            dynamics=self.denoising_model,
            in_node_nf=in_node_nf,
            n_dims=3,
            timesteps=timesteps,
            noise_schedule=noise_schedule,
            device=device,
        )

        # Save the Hyper-params used:
        self.save_hyperparameters()

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=24,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=24,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=24,
        )  # No need for shuffling - Will make visualisation easier.

    def configure_optimizers(self):
        """
        Setup Optimiser and learning rate scheduler if needed.
        """
        optimizer = torch.optim.Adam(
            self.diffusion_model.parameters(), lr=self.lr
        )  # noqa
        if self.learning_rate_schedule:
            lr_scheduler = {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.5,
                    patience=50,
                    min_lr=self.lr / 100,  # noqa
                ),
                "monitor": "val_loss",  # The metric to monitor
                "interval": "epoch",  # The interval to invoke the scheduler ('epoch' or 'step')  # noqa
                "frequency": 1,  # The frequency of scheduler invocation (every 1 epoch in this case)  # noqa
            }
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def forward(self, x, h, node_mask, edge_mask, edge_attributes):
        output = self.diffusion_model(
            x=x,
            h=h,
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_attributes=edge_attributes,
        )
        return output

    def training_step(self, batch, batch_idx):
        # Get Coordinates and node mask and edge attributes:
        coords, node_mask, edge_attributes = batch

        # Split the coords into H and X vectors:
        if self.no_product:
            h = coords[:, :, :-6]
        else:
            h = coords[:, :, :-3]
        x = coords[:, :, -3:]

        # Setup the Edge mask (1 everywhere except the diagonals - atom cannot be connected to itself):  # noqa
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = (
            ~torch.eye(edge_mask.size(-1), device=edge_mask.device)
            .unsqueeze(0)
            .bool()  # noqa
        )
        diag_mask = diag_mask.expand(edge_mask.size())
        edge_mask *= diag_mask

        # Forward pass:
        loss = self(x, h, node_mask, edge_mask, edge_attributes)

        # Log the loss:
        self.log(
            "train_loss", loss, on_epoch=True, prog_bar=True, on_step=False
        )  # noqa
        return loss

    def validation_step(self, batch, batch_idx):
        # Get Coordinates and node mask:
        coords, node_mask, edge_attributes = batch

        # Split the coords into H and X vectors:
        if self.no_product:
            h = coords[:, :, :-6]
        else:
            h = coords[:, :, :-3]
        x = coords[:, :, -3:]

        # Setup the Edge mask (1 everywhere except the diagonals - atom cannot be connected to itself):  # noqa
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = (
            ~torch.eye(edge_mask.size(-1), device=edge_mask.device)
            .unsqueeze(0)
            .bool()  # noqa
        )
        diag_mask = diag_mask.expand(edge_mask.size())
        edge_mask *= diag_mask

        # Forward pass:
        loss = self(x, h, node_mask, edge_mask, edge_attributes)

        # Log the loss:
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def sample_and_test(
        self,
        number_samples,
        true_h,
        true_x,
        node_mask,
        edge_mask,
        edge_attributes,
        folder_path,
        device=None,
    ):
        true_reactant = true_h[:, :, 4:7].clone()

        if not self.no_product:
            true_product = true_h[:, :, 7:10].clone()

        # Get the OHE of atom-type:
        atom_ohe = true_h[:, :, :4]

        # Inflate H so that it is the size of the number of samples:
        inflated_h = true_h.repeat(number_samples, 1, 1)

        # Inflate the node mask and edge masks:
        node_mask = node_mask.repeat(number_samples, 1)
        edge_mask = edge_mask.repeat(number_samples, 1, 1)

        # Inflate the Edge_Attributes:
        edge_attributes = edge_attributes.repeat(number_samples, 1)

        # Set model to evaluation mode (Faster computations and no data-leakage):  # noqa
        self.diffusion_model.eval()
        # Sample
        samples = self.diffusion_model.sample(
            inflated_h,
            edge_attributes,
            number_samples,
            23,
            node_mask.to(device),
            edge_mask.to(device),
        )

        # Round to prevent downstream type issues in RDKit:
        true_x = torch.round(true_x, decimals=3)

        # Concatenate the atom ohe with the true sample, reactant and product:
        true_sample = torch.cat([atom_ohe.to(device), true_x.to(device)], dim=2)  # noqa
        # Convert to XYZ format:
        true_samples = return_xyz(
            true_sample,
            ohe_dictionary=self.dataset.ohe_dict,
        )

        true_reactant = torch.cat(
            [atom_ohe.to(device), true_reactant.to(device)], dim=2
        )
        true_reactant = return_xyz(
            true_reactant,
            ohe_dictionary=self.dataset.ohe_dict,
        )

        if not self.no_product:
            true_product = torch.cat(
                [atom_ohe.to(device), true_product.to(device)], dim=2
            )  # noqa
            true_product = return_xyz(
                true_product,
                ohe_dictionary=self.dataset.ohe_dict,
            )

        # Save the true reactants/products/TS if save_samples set to true:
        if self.save_samples:
            true_filename = os.path.join(folder_path, "true_sample.xyz")
            write_xyz_file(true_samples, true_filename)

            reactant_filename = os.path.join(
                folder_path,
                "true_reactant.xyz",
            )
            write_xyz_file(true_reactant, reactant_filename)

            if not self.no_product:
                product_filename = os.path.join(
                    folder_path,
                    "true_product.xyz",
                )
                write_xyz_file(true_product, product_filename)

        for i in range(number_samples):
            predicted_sample = (
                samples[i].unsqueeze(0).to(torch.float64)
            )  # Unsqueeeze so it has bs

            # Need to round to make sure all the values are clipped and can be converted to doubles when using RDKit down the line:  # noqa
            predicted_sample = torch.round(predicted_sample, decimals=3)

            predicted_sample = torch.cat(
                [atom_ohe.to(device), predicted_sample], dim=2
            )  # noqa

            # Return it to xyz format:
            predicted_sample = return_xyz(
                predicted_sample,
                ohe_dictionary=self.dataset.ohe_dict,
            )

            # If the save samples is True then save it
            if self.save_samples:
                # Let's now try and save the molecule before aligning it and after to see the overlaps later on  # noqa
                aft_aligning_path = os.path.join(folder_path, f"sample_{i}.xyz")  # noqa

                # Save the samples:
                write_xyz_file(predicted_sample, aft_aligning_path)

    def test_step(self, batch, batch_idx):
        # Sample a bunch of test samples and then
        test_coords, test_node_mask, edge_attributes = batch

        # Setup the edge mask:
        test_edge_mask = test_node_mask.unsqueeze(1) * test_node_mask.unsqueeze(  # noqa
            2
        )  # noqa
        diag_mask = (
            ~torch.eye(test_edge_mask.size(-1), device=test_edge_mask.device)
            .unsqueeze(0)
            .bool()
        )
        diag_mask = diag_mask.expand(test_edge_mask.size())
        test_edge_mask *= diag_mask

        # If save_samples is true then make sure we have a folder for each samples:  # noqa
        if self.save_samples:
            self.save_path_batch = self.save_path + f"batch_{batch_idx}/"
            os.makedirs(self.save_path_batch, exist_ok=True)

        # Iterate over each test compound and create samples:
        for i in range(test_coords.shape[0]):
            # Create a subfolder for the molecule:
            if self.save_samples:
                self.save_path_mol = self.save_path_batch + f"mol_{i}/"
                os.makedirs(self.save_path_mol, exist_ok=True)
            else:
                self.save_path_mol = None

            # Seperate the Samples and then feed them through sampling method:
            # Split the coords into H and X:
            if self.no_product:
                test_h = test_coords[i, :, :-6].unsqueeze(0)
            else:
                test_h = test_coords[i, :, :-3].unsqueeze(0)

            test_x = test_coords[i, :, -3:].unsqueeze(0)
            node_mask_input = test_node_mask[i].unsqueeze(0)
            edge_mask_input = test_edge_mask[i].unsqueeze(0)

            edge_attribute_to_use = edge_attributes[i, :, :]

            # Create samples:
            self.sample_and_test(
                number_samples=self.test_sampling_number,
                true_h=test_h,
                true_x=test_x,
                node_mask=node_mask_input,
                edge_mask=edge_mask_input,
                edge_attributes=edge_attribute_to_use,
                folder_path=self.save_path_mol,
                device=self.device,
            )


if __name__ == "__main__":
    device = dynamics.setup_device()

    # Assign which dataset to use:
    dataset_to_use = "RGD1"

    # Use Graph Model or not?
    use_reaction_graph_model = False

    # Setup Hyper-paremetres:
    learning_rate_schedule = False
    random_rotations = False  # Part of Data Augmentation
    augment_train_set = False  # Also part of Data Augmentation
    remove_hydrogens = True  # Only Possible with the W93 Dataset
    include_context = (
        None  # "Activation_Energy"  # Only Possible with the W93 Dataset # noqa
    )

    # # If we do not include the product in the diffusoin step:
    no_product = True
    in_edge_nf = 2  # When we have the product in the graph

    in_node_nf = get_node_features(
        remove_hydrogens=remove_hydrogens,
        include_context=include_context,
        no_product=no_product,
    )
    noise_schedule = "sigmoid_2"  # "sigmoid_INTEGER"
    timesteps = 1_000
    batch_size = 294
    n_layers = 8
    hidden_features = 64
    lr = 1e-4
    epochs = 2_000

    # Setup Saving path:
    model_name = f"{no_product}_no_product_{use_reaction_graph_model}_graph_model_{dataset_to_use}_dataset_{include_context}_context_{random_rotations}_Random_rotations_{augment_train_set}_augment_train_set_{n_layers}_layers_{hidden_features}_hiddenfeatures_{lr}_lr_{noise_schedule}_{timesteps}_timesteps_{batch_size}_batch_size_{epochs}_epochs_{remove_hydrogens}_Rem_Hydrogens"  # noqa
    folder_name = (
        f"src/Diffusion/{dataset_to_use}TESTING_FAKE_dataset_weights/"
        + model_name
        + "/"
    )  # noqa

    # Create the directories:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    model_path = folder_name + "Weights/"
    sample_path = folder_name + "Samples/"

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    if not use_reaction_graph_model:
        # Setup model:
        lit_diff_model = LitDiffusionModel(
            dataset_to_use=dataset_to_use,
            in_node_nf=in_node_nf,
            hidden_features=hidden_features,
            n_layers=n_layers,
            device=device,
            lr=lr,
            remove_hydrogens=remove_hydrogens,
            test_sampling_number=1,
            save_samples=False,
            save_path=None,
            timesteps=timesteps,
            noise_schedule=noise_schedule,
            random_rotations=random_rotations,
            augment_train_set=augment_train_set,
            include_context=include_context,
            learning_rate_schedule=learning_rate_schedule,
            no_product=no_product,
            batch_size=batch_size,
        )
    else:
        # Setup Graph Diffusion model:
        lit_diff_model = LitDiffusionModel_With_graph(
            in_node_nf=in_node_nf,
            in_edge_nf=in_edge_nf,
            hidden_features=hidden_features,
            n_layers=n_layers,
            device=device,
            lr=lr,
            test_sampling_number=1,
            save_samples=False,
            save_path=None,
            timesteps=timesteps,
            noise_schedule=noise_schedule,
            learning_rate_schedule=learning_rate_schedule,
            no_product=no_product,
            batch_size=batch_size,
        )

    # # Load the weights from initial state:
    # path_to_load = "src/Diffusion/W93_dataset_weights/False_no_productW93_dataset_Activation_Energy_context_False_Random_rotations_True_augment_train_set_8_layers_64_hiddenfeatures_0.0001_lr_sigmoid_2_1000_timesteps_64_batch_size_2000_epochs_False_Rem_Hydrogens/Weights/weights.pth"  # noqa
    # lit_diff_model.load_state_dict(torch.load(path_to_load))

    # Create WandB logger:
    wandb_logger = pytorch_lightning.loggers.WandbLogger(
        project="Diffusion_1234testingsetup",
        name=model_name,  # Diffusion_large_dataset
    )

    # Setup a learning rate monitor that prints to WandB when we use a learning rate scheduler:  # noqa
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Train
    trainer = pl.Trainer(
        accelerator="cuda",
        max_epochs=epochs,
        logger=wandb_logger,
        callbacks=[lr_monitor],
        fast_dev_run=False,
    )

    trainer.fit(lit_diff_model)

    # Add filename:
    model_path = os.path.join(model_path, "weights.pth")

    torch.save(lit_diff_model.state_dict(), model_path)
    wandb.finish()
