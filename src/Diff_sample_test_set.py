"""
This is the script to evaluate our diffusion model using python Lightning
-------------------------------------------------------------------------
"""
import pytorch_lightning as pl
from Diffusion.Equivariant_Diffusion import *
from EGNN import model_dynamics_with_mask
from sklearn.model_selection import train_test_split

from Dataset_W93.dataset_class import * 

from Diff_lightning import LitDiffusionModel



def test_model(model, logger):
    """
    Function to test the Lightning model
    """
    trainer = pl.Trainer(accelerator='cuda', logger=logger, fast_dev_run=False)
    trainer.test(model)


if __name__ == "__main__":
    
    # Hyper-parameters: 
    device = model_dynamics_with_mask.setup_device()


    dataset_to_use = "TX1"

    # Setup Hyper-paremetres:
    learning_rate_schedule = False
    random_rotations = False     # Part of Data Augmentation
    augment_train_set = False   # Also part of Data Augmentation
    remove_hydrogens = False
    include_context = False
            
    if remove_hydrogens:
        in_node_nf = 9 + 1  # To account for time and 1 less OHE
    else:
        in_node_nf = 10 + 1  # To account for time
    
    if include_context:
        in_node_nf += 1     # Add one for the size of context --> For now we just have the Nuclear Charge
    
    
    # src/Diffusion/Clean_lightning/True_for_W93_DATASET_5_layers_64_hiddenfeatures_0.0001_lr_cosine_2000_timesteps_64_batch_size_1000_epochs_False_Rem_Hydrogens/Samples_test
    out_node = 3
    context_nf = 0 
    n_dims = 3
    noise_schedule = "sigmoid_2"
    loss_type = "l2"
    timesteps = 2_000
    batch_size = 64
    n_layers = 8
    hidden_features = 64
    lr = 1e-4
    epochs = 1000
    test_sampling_number=40
    save_samples = True
    save_path = "src/Diffusion/Clean_lightning/TX1_dataset_False_include_VAN_DER_WAAL_RADII_False_Random_rotations_False_augment_train_set_8_layers_64_hiddenfeatures_0.0001_lr_sigmoid_2_2000_timesteps_64_batch_size_3000_epochs_False_Rem_Hydrogens/Samples/"


    # Create an instance of your Lightning model
    lit_diff_model = LitDiffusionModel(dataset_to_use,
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
                                        random_rotations=random_rotations,
                                        augment_train_set=augment_train_set,
                                        include_context=include_context, 
                                        learning_rate_schedule = learning_rate_schedule)
        
    print("Model parameters device:", next(lit_diff_model.parameters()).device)


    # Load the saved model state dictionary
    model_path = "src/Diffusion/Clean_lightning/TX1_dataset_False_include_VAN_DER_WAAL_RADII_False_Random_rotations_False_augment_train_set_8_layers_64_hiddenfeatures_0.0001_lr_sigmoid_2_2000_timesteps_64_batch_size_3000_epochs_False_Rem_Hydrogens/Weights/weights.pth"
    

        
    # Load the state dict into the model:
    lit_diff_model.load_state_dict(torch.load(model_path))

    # Create a trainer instance for testing
    test_model(lit_diff_model, logger = None)