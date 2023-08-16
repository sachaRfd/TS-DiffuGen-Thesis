"""
Script to train our diffusion model using python Lightning
----------------------------------------------------------
"""

import os
import pytorch_lightning
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import wandb


from Diffusion_graph.Diffusion_product_graph import DiffusionModel
from src_coords_graphs.EGNN_product_graph.dynamics_with_graph import (
    EGNN_dynamics,
)
from src.Diffusion.saving_sampling_functions import write_xyz_file, return_xyz
import src.Diffusion.utils as Diffusion_utils
from src.Diffusion.utils import random_rotation
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau


from data.Dataset_W93.dataset_reactant_and_product_graph import (
    QM90_TS_reactant_coords_and_product_graph,
)
from data.Dataset_TX1.dataset_TX1_class import TX1_dataset

from pytorch_lightning.callbacks import LearningRateMonitor


# Pytorch Lightning class for the diffusion model:
class LitDiffusionModel(pl.LightningModule):
    def __init__(
        self,
        dataset_to_use,
        in_node_nf,
        in_edge_nf,
        context_nf,
        hidden_features,
        out_node,
        n_dims,
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
        use_product_graph=False,
        no_product=False,
    ):
        super(LitDiffusionModel, self).__init__()
        self.dataset_to_use = dataset_to_use
        self.batch_size = 64
        self.lr = lr
        self.remove_hydrogens = remove_hydrogens
        self.include_context = include_context
        self.learning_rate_schedule = learning_rate_schedule
        self.use_product_graph = use_product_graph

        self.no_product = no_product
        print(f"There are {in_node_nf} features per node")

        # Assert that product grpah can only be used with the W93 Dataset, with no context and with hydrogens:      # noqa
        if self.use_product_graph:
            assert (
                not self.include_context
            ), "For the TX1 dataset, including context is not allowed."
            assert (
                not self.remove_hydrogens
            ), "For the TX1 dataset, removing hydrogens is not allowed."
            assert (
                self.dataset_to_use == "W93"
            ), "Using product graphs is only currently available with the W93 Dataset"  # noqa
            assert (
                not augment_train_set
            ), "Augmenting the dataset is not possible when using products as graphs"  # noqa

        # Assert that if are not using products then we will not be able to do train augmentations      # noqa
        if self.no_product:
            assert (
                not augment_train_set
            ), "Cannot augment the train set if the products are not used"

        # Assert you have the correct datasets:
        assert (
            self.dataset_to_use == "W93" or self.dataset_to_use == "TX1"
        ), "Dataset can only be W93 for TS-DIFF dataset or TX1 for the MIT paper"  # noqa

        # Add asserts for what is possible with the TX1 dataset and what is possible with the W93 dataset  # noqa
        if self.dataset_to_use == "TX1":
            assert (
                not self.include_context
            ), "For the TX1 dataset, including context is not allowed."
            assert (
                not self.remove_hydrogens
            ), "For the TX1 dataset, removing hydrogens is not allowed."

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

        if self.dataset_to_use == "W93" and not self.use_product_graph:
            self.dataset = QM90_TS_reactant_coords_and_product_graph(
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

        elif self.dataset_to_use == "W93" and self.use_product_graph:
            self.dataset = QM90_TS_reactant_coords_and_product_graph(
                graph_product=self.use_product_graph,
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

        # Augment the train set here:
        if self.augment_train_set:
            self.augment_reactants_products()

        # Setup the denoising model:
        self.denoising_model = EGNN_dynamics(
            in_node_nf=in_node_nf,
            context_node_nf=context_nf,
            hidden_nf=hidden_features,
            in_edge_nf=in_edge_nf,
            out_node=out_node,
            n_dims=n_dims,
            sin_embedding=True,
            n_layers=n_layers,
            device=device,
            attention=False,
        )

        # Setup the diffusion model:
        self.diffusion_model = DiffusionModel(
            dynamics=self.denoising_model,
            in_node_nf=in_node_nf,
            n_dims=n_dims,
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
                    patience=75,
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
        # if self.use_product_graph:
        # Get Coordinates and node mask and edge attributes:
        coords, node_mask, edge_attributes = batch
        # else:
        #     # Get Coordinates and node masks only:
        #     coords, node_mask = batch

        # Split the coords into H and X vectors:
        if self.no_product:
            h = coords[:, :, :-9]
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
            h = coords[:, :, :-9]
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
        remove_hydrogens=False,
        device=None,
    ):
        # Get the true reactant and product - Still hard-coded:
        if remove_hydrogens:
            true_reactant = true_h[
                :, :, 3 + self.context_size : 6 + self.context_size  # noqa
            ].clone()
            # true_product = true_h[
            #     :, :, 6 + self.context_size : 9 + self.context_size  # noqa
            # ].clone()

            # Get the OHE of atom-type:
            atom_ohe = true_h[:, :, :3]

            # Split the H depending on if we want products to be included or not:   # noqa
            # if self.no_product:
            #     true_h = true_h[:, :, :-6]
            # else:
            #     true_h = true_h[:, :, :-3]

        else:
            true_reactant = true_h[
                :, :, 4 + self.context_size : 7 + self.context_size  # noqa
            ].clone()

            # true_product = true_h[
            #     :, :, 7 + self.context_size : 10 + self.context_size  # noqa
            # ].clone()

            # Get the OHE of atom-type:
            atom_ohe = true_h[:, :, :4]

            # Split the H depending on if we want products to be included or not:   # noqa
            # if self.no_product:
            #     true_h = true_h[:, :, :-6]
            # else:
            #     true_h = true_h[:, :, :-3]

        # Inflate H so that it is the size of the number of samples:
        inflated_h = true_h.repeat(number_samples, 1, 1)

        # Inflate the node mask and edge masks:
        node_mask = node_mask.repeat(number_samples, 1)
        edge_mask = edge_mask.repeat(number_samples, 1, 1)

        # Inflate the Edge_Attributes:
        edge_attributes = edge_attributes.repeat(number_samples, 1)

        # Set model to evaluation mode (Faster computations and no data-leakage):  # noqa
        self.diffusion_model.eval()
        if remove_hydrogens:
            samples = self.diffusion_model.sample(
                inflated_h,
                edge_attributes,
                number_samples,
                7,
                node_mask.to(device),
                edge_mask.to(device),
                context_size=self.context_size,
            )
        else:
            samples = self.diffusion_model.sample(
                inflated_h,
                edge_attributes,
                number_samples,
                23,
                node_mask.to(device),
                edge_mask.to(device),
                context_size=self.context_size,
            )

        # Round to prevent downstream type issues in RDKit:
        true_x = torch.round(true_x, decimals=3)

        # Concatenate the atom ohe with the true sample, reactant and product:
        true_sample = torch.cat([atom_ohe.to(device), true_x.to(device)], dim=2)  # noqa
        true_reactant = torch.cat(
            [atom_ohe.to(device), true_reactant.to(device)], dim=2
        )
        # true_product = torch.cat(
        #     [atom_ohe.to(device), true_product.to(device)], dim=2
        # )  # noqa

        # Convert to XYZ format:
        true_samples = return_xyz(
            true_sample, dataset=self.dataset, remove_hydrogen=remove_hydrogens
        )
        true_reactant = return_xyz(
            true_reactant,
            dataset=self.dataset,
            remove_hydrogen=remove_hydrogens,  # noqa
        )

        # true_product = return_xyz(
        #     true_product,
        #     dataset=self.dataset,
        #     remove_hydrogen=remove_hydrogens,  # noqa
        # )

        # Save the true reactants/products/TS if save_samples set to true:
        if self.save_samples:
            true_filename = os.path.join(folder_path, "true_sample.xyz")
            write_xyz_file(true_samples, true_filename)

            reactant_filename = os.path.join(folder_path, "true_reactant.xyz")
            write_xyz_file(true_reactant, reactant_filename)

            # product_filename = os.path.join(folder_path, "true_product.xyz")
            # write_xyz_file(true_product[0], product_filename)

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
                dataset=self.dataset,
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
            test_h = test_coords[i, :, :-3].unsqueeze(
                0
            )  # Changed from -3 to -6 because we dont want to include the product info    # noqa
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
                remove_hydrogens=self.remove_hydrogens,
                device=self.device,
            )


if __name__ == "__main__":
    device = Diffusion_utils.setup_device()

    # Assign which dataset to use:
    dataset_to_use = "W93"
    use_product_graph = True
    no_product = True

    # Setup Hyper-paremetres:
    learning_rate_schedule = False
    random_rotations = False  # Part of Data Augmentation
    augment_train_set = False  # Also part of Data Augmentation
    remove_hydrogens = False  # Only Possible with the W93 Dataset
    include_context = False  # Only Possible with the W93 Dataset

    in_edge_nf = 2  # When we have the product in the graph

    if remove_hydrogens:
        in_node_nf = 9 + 1  # To account for time and 1 less OHE
    else:
        in_node_nf = 10 + 1  # To account for time

    if no_product:
        in_node_nf -= 6

    if include_context:
        in_node_nf += 1  # Add one for the size of context --> For now we just have the Nuclear Charge  # noqa

    out_node = 3
    context_nf = 0
    n_dims = 3
    noise_schedule = "sigmoid_2"
    timesteps = 1_000
    batch_size = 64
    n_layers = 8
    hidden_features = 64
    lr = 1e-4
    epochs = 3_000

    # Setup Saving path:
    model_name = f"only_GRAPH_{use_product_graph}_use_product_graph_{dataset_to_use}_dataset_{include_context}_include_VAN_DER_WAAL_RADII_{random_rotations}_Random_rotations_{augment_train_set}_augment_train_set_{n_layers}_layers_{hidden_features}_hiddenfeatures_{lr}_lr_{noise_schedule}_{timesteps}_timesteps_{batch_size}_batch_size_{epochs}_epochs_{remove_hydrogens}_Rem_Hydrogens_second"  # noqa
    folder_name = (
        "src_coords_graphs/Diffusion/weights_and_samples/" + model_name + "/"
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

    # Setup model:
    lit_diff_model = LitDiffusionModel(
        dataset_to_use=dataset_to_use,
        in_node_nf=in_node_nf,
        context_nf=context_nf,
        in_edge_nf=in_edge_nf,
        hidden_features=hidden_features,
        out_node=out_node,
        n_dims=n_dims,
        n_layers=n_layers,
        device=device,
        lr=lr,
        remove_hydrogens=remove_hydrogens,
        test_sampling_number=1,
        save_samples=False,
        save_path=sample_path,
        timesteps=timesteps,
        noise_schedule=noise_schedule,
        random_rotations=random_rotations,
        augment_train_set=augment_train_set,
        include_context=include_context,
        learning_rate_schedule=learning_rate_schedule,
        use_product_graph=use_product_graph,
        no_product=no_product,
    )

    # path_to_load = "src_coords_graphs/Diffusion/weights_and_samples/only_reactant_True_use_product_graph_W93_dataset_False_include_VAN_DER_WAAL_RADII_False_Random_rotations_False_augment_train_set_8_layers_64_hiddenfeatures_0.0001_lr_sigmoid_2_1000_timesteps_64_batch_size_2000_epochs_False_Rem_Hydrogens/Weights/weights.pth"  # noqa
    # lit_diff_model.load_state_dict(torch.load(path_to_load))

    # Create WandB logger:
    wandb_logger = pytorch_lightning.loggers.WandbLogger(
        project="Diffusion_graph_experiment", name=model_name
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
