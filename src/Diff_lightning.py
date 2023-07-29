"""
Script to train our diffusion model using python Lightning
----------------------------------------------------------
"""

import os
import pytorch_lightning
import pytorch_lightning as pl

from Diffusion.Equivariant_Diffusion import *
from Diffusion.saving_sampling_functions import write_xyz_file, return_xyz
from Diffusion.utils import random_rotation
from EGNN import model_dynamics_with_mask
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau


from Dataset_W93.dataset_class import QM90_TS
from Dataset_TX1.dataset_TX1_class import TX1_dataset

from pytorch_lightning.callbacks import LearningRateMonitor


# Pytorch Lightning class for the diffusion model:
class LitDiffusionModel(pl.LightningModule):
    def __init__(self, 
                 dataset_to_use,
                 in_node_nf,
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
                 include_context = False,
                 learning_rate_schedule=False):
        super(LitDiffusionModel, self).__init__()
        self.dataset_to_use = dataset_to_use
        self.batch_size = 64
        self.lr = lr
        self.remove_hydrogens = remove_hydrogens
        self.include_context = include_context
        self.learning_rate_schedule = learning_rate_schedule



        # Assert you have the correct datasets: 
        assert self.dataset_to_use == "W93" or self.dataset_to_use == "TX1", "Dataset can only be W93 for TS-DIFF dataset or TX1 for the MIT paper"

        # Add asserts for what is possible with the TX1 dataset and what is possible with the W93 dataset
        if self.dataset_to_use == "TX1":
            assert not self.include_context, "For the TX1 dataset, including context is not allowed."
            assert not self.remove_hydrogens, "For the TX1 dataset, removing hydrogens is not allowed."

        # For now, context is only 1 variable added to the node features - Only possible with the W93 dataset - Try and not make this hard-coded
        if self.include_context and self.dataset_to_use == "W93":
            self.context_size = 1
        else:
            self.context_size = 0
                    
        # Variables for testing (Number of samples and if we want to save the samples): 
        self.test_sampling_number = test_sampling_number
        self.save_samples = save_samples
        

        # Include Random Rotations of the molecules so that it is different at each epoch: 
        self.data_random_rotations = random_rotations

        # Augment train set by duplicating train reactions but reversing reactants and products with each-other:
        self.augment_train_set = augment_train_set


        if self.save_samples:
            # assert that the save_path input is correct:
            assert os.path.exists(save_path)
            self.save_path = save_path


        if self.dataset_to_use == "W93":        
            self.dataset = QM90_TS(directory="data/Dataset_W93/data/Clean_Geometries", remove_hydrogens=self.remove_hydrogens, include_context=include_context)
        
            # # Split into 8:1:1 ratio:
            self.train_dataset, test_dataset = train_test_split(self.dataset, test_size=0.2, random_state=42)
            self.val_dataset, self.test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=42)

        
        elif self.dataset_to_use == "TX1":
            self.dataset =  TX1_dataset(split="data")

            # We want to match the sizes used in MIT paper 9,000 for training and 1,073 for testing
            train_dataset, self.test_dataset = train_test_split(self.dataset, test_size=0.1, random_state=42, shuffle=True)
            
            # Split the training set into a 8:1 split to train/val set: 
            self.train_dataset, self.val_dataset = train_test_split(train_dataset, test_size=1/9, random_state=42, shuffle=True)



        # Augment the train set here: 
        if self.augment_train_set:
            self.augment_reactants_products()

        # Setup the denoising model: 
        self.denoising_model = model_dynamics_with_mask.EGNN_dynamics_QM9(in_node_nf=in_node_nf, context_node_nf=context_nf, hidden_nf=hidden_features, out_node=out_node, n_dims=n_dims, sin_embedding=True, n_layers=n_layers, device=device, attention=False)
        
        # Setup the diffusion model:
        self.diffusion_model = DiffusionModel(dynamics=self.denoising_model, in_node_nf=in_node_nf, n_dims=n_dims, timesteps=timesteps, noise_schedule=noise_schedule, device=device)

        # Save the Hyper-params used: 
        self.save_hyperparameters()

    def augment_reactants_products(self):
        """
        This function will augment the train set
        """
        print("Augmenting the train set, by duplicating train reactions and replacing reactants with products and vice-versa")
        train_set_augmented = []


        for data in self.train_dataset:
            train_set_augmented.append(data)  # Append the original data
            # Seperate into graphs and node masks
            molecule, node_mask = data[0], data[1]

            if self.remove_hydrogens:
                # Swap the reactant and product - but keep the OHE
                swapped_molecule = torch.cat((molecule[:, :3 + self.context_size], molecule[:, 6 + self.context_size:9+self.context_size], molecule[:, 3+self.context_size:6+self.context_size], molecule[:, 9+self.context_size:]), dim=1)

            else:
                swapped_molecule = torch.cat((molecule[:, :4 + self.context_size], molecule[:, 7+self.context_size:10+self.context_size], molecule[:, 4+self.context_size:7+self.context_size], molecule[:, 10+self.context_size:]), dim=1)
 
            # Now we can create the tuple with the node mask (No Need to swap anything within the node_mask)
            swapped_data = (swapped_molecule, node_mask)
            # Append it to the train set: 
            train_set_augmented.append(swapped_data)
        
        # Swap the train set with the augmented one: 
        self.train_dataset = train_set_augmented



    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=24) 
    
    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=24)
    
    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=24)     # No need for shuffling - Will make visualisation easier. 
    
    def configure_optimizers(self):
        """
        Setup Optimiser and learning rate scheduler if needed. 
        """
        optimizer = torch.optim.Adam(self.diffusion_model.parameters(), lr=self.lr)
        if self.learning_rate_schedule: 
            lr_scheduler = {
                'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=75, min_lr=self.lr / 100),
                'monitor': 'val_loss',  # The metric to monitor
                'interval': 'epoch',  # The interval to invoke the scheduler ('epoch' or 'step')
                'frequency': 1  # The frequency of scheduler invocation (every 1 epoch in this case)
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
        h = coords[:, :, :-3]
        x = coords[:, :, -3:]

        # Setup the Edge mask (1 everywhere except the diagonals - atom cannot be connected to itself):
        edge_mask = node_mask.unsqueeze(1)  * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(-1), device=edge_mask.device).unsqueeze(0).bool()
        diag_mask = diag_mask.expand(edge_mask.size())
        edge_mask *= diag_mask

        # Use random rotations during training if required: 
        if self.data_random_rotations: 
            x, h  = random_rotation(x, h)


        # Forward pass: 
        loss = self(x, h, node_mask, edge_mask)  

        # Log the loss: 
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=False)  
        return loss
    

    def validation_step(self, batch, batch_idx):
        
        # Get Coordinates and node mask:
        coords, node_mask = batch

        # Split the coords into H and X:
        h = coords[:, :, :-3]
        x = coords[:, :, -3:]

        # Setup the Edge mask (1 everywhere except the diagonals - atom cannot be connected to itself):
        edge_mask = node_mask.unsqueeze(1)  * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(-1), device=edge_mask.device).unsqueeze(0).bool()
        diag_mask = diag_mask.expand(edge_mask.size())
        edge_mask *= diag_mask

        # Forward pass: 
        loss = self(x, h, node_mask, edge_mask)  

        # Log the loss: 
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)     
        return loss
    
    def sample_and_test(self, number_samples, true_h, true_x, node_mask, edge_mask, folder_path, remove_hydrogens=False, device=None):
        
        # Get the true reactant and product - Still hard-coded: 
        if remove_hydrogens:
            true_reactant = true_h[:, :, 3 + self.context_size:6 + self.context_size].clone()
            true_product = true_h[:, :, 6 + self.context_size:9 + self.context_size].clone()

            # Get the OHE of atom-type: 
            atom_ohe = true_h[:, :, :3]
        
        else: 
            true_reactant = true_h[:, :, 4 + self.context_size:7 + self.context_size].clone()
            true_product = true_h[:, :, 7 + self.context_size:10 +self.context_size].clone()
            
            # Get the OHE of atom-type:  
            atom_ohe = true_h[:, :, :4]

        # Inflate H so that it is the size of the number of samples:
        inflated_h = true_h.repeat(number_samples, 1, 1)

        # Inflate the node mask and edge masks: 
        node_mask = node_mask.repeat(number_samples, 1)
        edge_mask = edge_mask.repeat(number_samples, 1, 1)

        # Set model to evaluation mode (Faster computations and no data-leakage):
        self.diffusion_model.eval()
        if remove_hydrogens:
            samples  = self.diffusion_model.sample(inflated_h, number_samples, 7, node_mask.to(device), edge_mask.to(device), context_size=self.context_size)   # still hard coded
        else: 
            samples  = self.diffusion_model.sample(inflated_h, number_samples, 23, node_mask.to(device), edge_mask.to(device), context_size=self.context_size)  # Still hard coded

        # Round to prevent downstream type issues in RDKit:
        true_x = torch.round(true_x, decimals=3)

        # Concatenate the atom ohe with the true sample, reactant and product:
        true_sample = torch.cat([atom_ohe.to(device), true_x.to(device)], dim=2)
        true_reactant = torch.cat([atom_ohe.to(device), true_reactant.to(device)], dim=2)
        true_product = torch.cat([atom_ohe.to(device), true_product.to(device)], dim=2)

        # Convert to XYZ format: 
        true_samples = return_xyz(true_sample, dataset=self.dataset, remove_hydrogen=remove_hydrogens)
        true_reactant = return_xyz(true_reactant, dataset=self.dataset, remove_hydrogen=remove_hydrogens)
        true_product = return_xyz(true_product, dataset=self.dataset, remove_hydrogen=remove_hydrogens)


        # Save the true reactants/products/TS if save_samples set to true:
        if self.save_samples:
            true_filename = os.path.join(folder_path, "true_sample.xyz")
            write_xyz_file(true_samples[0], true_filename)

            reactant_filename = os.path.join(folder_path, "true_reactant.xyz")
            write_xyz_file(true_reactant[0], reactant_filename)

            product_filename = os.path.join(folder_path, "true_product.xyz")
            write_xyz_file(true_product[0], product_filename)

        
        for i in range(number_samples):
            predicted_sample = samples[i].unsqueeze(0).to(torch.float64)  # Unsqueeeze so it has bs


            # Need to round to make sure all the values are clipped and can be converted to doubles when using RDKit down the line: 
            predicted_sample = torch.round(predicted_sample, decimals=3)

            predicted_sample = torch.cat([atom_ohe.to(device), predicted_sample], dim=2)
            
            # Return it to xyz format: 
            predicted_sample = return_xyz(predicted_sample, dataset=self.dataset, remove_hydrogen=remove_hydrogens)


            # If the save samples is True then save it
            if self.save_samples:               
                # Let's now try and save the molecule before aligning it and after to see the overlaps later on
                aft_aligning_path = os.path.join(folder_path, f"sample_{i}.xyz")

                # Save the samples:
                write_xyz_file(predicted_sample[0], aft_aligning_path) 



    def test_step(self, batch, batch_idx):
        # Sample a bunch of test samples and then
        test_coords, test_node_mask = batch

        # Setup the edge mask:
        test_edge_mask = test_node_mask.unsqueeze(1)  * test_node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(test_edge_mask.size(-1), device=test_edge_mask.device).unsqueeze(0).bool()
        diag_mask = diag_mask.expand(test_edge_mask.size())
        test_edge_mask *= diag_mask


        # If save_samples is true then make sure we have a folder for each samples: 
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
            test_h = test_coords[i, :, :-3].unsqueeze(0)
            test_x = test_coords[i, :, -3:].unsqueeze(0)
            node_mask_input = test_node_mask[i].unsqueeze(0)
            edge_mask_input = test_edge_mask[i].unsqueeze(0)

            # Create samples: 
            self.sample_and_test(self.test_sampling_number, test_h, test_x, node_mask_input, edge_mask_input, self.save_path_mol, self.remove_hydrogens, self.device)


if __name__ == "__main__":

    device = model_dynamics_with_mask.setup_device()
    
    # Assign which dataset to use: 
    dataset_to_use = "TX1"

    # Setup Hyper-paremetres:
    learning_rate_schedule = False
    random_rotations = False     # Part of Data Augmentation    
    augment_train_set = False   # Also part of Data Augmentation    
    remove_hydrogens = False    # Only Possible with the W93 Dataset
    include_context = False     # Only Possible with the W93 Dataset

    if remove_hydrogens:
        in_node_nf = 9 + 1  # To account for time and 1 less OHE
    else:
        in_node_nf = 10 + 1  # To account for time
    
    if include_context:
        in_node_nf += 1     # Add one for the size of context --> For now we just have the Nuclear Charge
    
    
    out_node = 3
    context_nf = 0 
    n_dims = 3
    noise_schedule = "sigmoid_2"   # "sigmoid_INTEGER"
    timesteps = 2_000
    batch_size = 64
    n_layers = 8
    hidden_features = 64
    lr = 1e-4
    epochs = 3000

    # Setup Saving path: 
    model_name = f"{dataset_to_use}_dataset_{include_context}_include_VAN_DER_WAAL_RADII_{random_rotations}_Random_rotations_{augment_train_set}_augment_train_set_{n_layers}_layers_{hidden_features}_hiddenfeatures_{lr}_lr_{noise_schedule}_{timesteps}_timesteps_{batch_size}_batch_size_{epochs}_epochs_{remove_hydrogens}_Rem_Hydrogens"
    folder_name = "src/Diffusion/Clean_lightning/" + model_name + "/"

    # Create the directories: 
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    model_path = folder_name + f"Weights/"
    sample_path = folder_name + "Samples/"
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if not os.path.exists(sample_path):
        os.makedirs(sample_path)



    # Setup model:
    lit_diff_model = LitDiffusionModel(dataset_to_use=dataset_to_use,
                                       in_node_nf=in_node_nf,
                                       context_nf=context_nf, 
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
                                       learning_rate_schedule = learning_rate_schedule)
    

    # Create WandB logger:    
    wandb_logger = pytorch_lightning.loggers.WandbLogger(project='Diffusion_5_layer_2000_timesteps', name=model_name)
    
    # Setup a learning rate monitor that prints to WandB when we use a learning rate scheduler: 
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Train
    trainer = pl.Trainer(accelerator="cuda",
                         max_epochs=epochs, 
                         logger=wandb_logger, 
                         callbacks=[lr_monitor], 
                         fast_dev_run=False)
    
    trainer.fit(lit_diff_model)

    # Add filename: 
    model_path = os.path.join(model_path, f"weights.pth")

    torch.save(lit_diff_model.state_dict(), model_path)
    wandb.finish()


    
