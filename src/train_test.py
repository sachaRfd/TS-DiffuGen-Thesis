# Sacha Raffaud sachaRfd and acse-sr1022

import argparse
import yaml
import os

import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor


from src.lightning_setup import LitDiffusionModel, LitDiffusionModel_With_graph
from src.Diffusion.equivariant_diffusion import get_node_features

from src.EGNN.utils import setup_device


"""

Before training a diffusion model, please make sure your python environment
initialises WandB accordingly. 

Setting up WandB for smooth training can be done as by inserting your WandB 
API key inside the wandb_setup.py script located in the root directory and 
then running that script.


This file contains the function for training and testing the diffusion models.

The sampling/testing framework has a test within the PyTest framework, however,
the training script does not. This is because it requires access to WandB to log
the losses for model managment.



"""  # noqa


def main(args, pytest_time=False):
    """
    Main function to manage training and testing of the diffusion model.

    This function serves as the entry point to the script, allowing for both training and
    testing modes based on the provided command-line arguments. It sets up the necessary
    configurations, loads or trains the model, and handles the flow accordingly.

    Args:
        args (argparse.Namespace): Command-line arguments parsed using argparse.
        pytest_time (bool, optional): Flag indicating whether the code is being run in a pytest environment (default is False).

    Returns:
        None
    """  # noqa
    # Get Variables:
    # In_Nodes:
    in_node_nf = get_node_features(
        remove_hydrogens=args.remove_hydrogens,
        include_context=args.include_context,
        no_product=args.remove_product,
    )
    # At pytest time: Device should be the CPU
    if pytest_time:
        device = "cpu"
    else:
        # Setup the device that is available:
        device = setup_device()

    # Get the paths:
    model_name = args.model_name
    folder_name = args.folder_name
    folder_name = "trained_models/" + folder_name
    # Create the Weights and Samples folder:
    model_path = folder_name + "/Weights/"
    sample_path = folder_name + "/Samples/"
    model_path_saved = os.path.join(model_path, "weights.pth")

    if args.train_test == "train":
        # If in training mode - Check that the path is not already present:
        assert (
            folder_name not in os.listdir()
        ), f"{folder_name} Folder already exists"  # noqa

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if not os.path.exists(sample_path):
            os.makedirs(sample_path)

        # Check if user wants to use graphs:
        if args.use_graph_in_model:
            # Setup Graph Diffusion model:
            lit_diff_model = LitDiffusionModel_With_graph(
                in_node_nf=in_node_nf,
                in_edge_nf=2,
                hidden_features=args.hidden_features,
                n_layers=args.n_layers,
                device=device,
                lr=args.lr,
                test_sampling_number=None,
                save_samples=False,
                save_path=None,
                timesteps=args.timesteps,
                noise_schedule=args.noise_schedule,
                learning_rate_schedule=args.learning_rate_scheduler,
                no_product=args.remove_product,
                batch_size=args.batch_size,
                pytest_time=pytest_time,  # ADDED HERE CHECK IT WORKS
            )

        else:
            # Setup regular model:
            lit_diff_model = LitDiffusionModel(
                dataset_to_use=args.dataset_to_use,
                in_node_nf=in_node_nf,
                hidden_features=args.hidden_features,
                n_layers=args.n_layers,
                device=device,
                lr=args.lr,
                remove_hydrogens=args.remove_hydrogens,
                test_sampling_number=None,
                save_samples=False,
                save_path=None,
                timesteps=args.timesteps,
                noise_schedule=args.noise_schedule,
                random_rotations=args.random_rotations,
                augment_train_set=args.augment_train_set,
                include_context=args.include_context,
                learning_rate_schedule=args.learning_rate_scheduler,
                no_product=args.remove_product,
                batch_size=args.batch_size,
                pytest_time=pytest_time,  # ADDED HERE CHECK IT WORKS
            )

        # Setup the WandB Logger:
        wandb_logger = pl.loggers.WandbLogger(
            project=args.wandb_project_name,
            name=model_name,
        )

        # Setup LR logger:
        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        # Learning rate scheduler:
        if args.learning_rate_scheduler:
            trainer = pl.Trainer(
                accelerator="cuda",
                max_epochs=args.epochs,
                logger=wandb_logger,
                callbacks=[lr_monitor],
                fast_dev_run=False,
            )  # Fast develop run was used for debugging purposes

            trainer.fit(lit_diff_model)

        else:
            trainer = pl.Trainer(
                accelerator="cuda",
                max_epochs=args.epochs,
                logger=wandb_logger,
                fast_dev_run=False,
            )

            trainer.fit(lit_diff_model)

        # Save the model once training has finished:
        torch.save(lit_diff_model.state_dict(), model_path_saved)
        wandb.finish()

    elif args.train_test == "test":
        # Check that the folders exist:
        assert os.path.exists(
            folder_name
        ), f"{folder_name} Folder Path is Incorrect"  # noqa
        assert os.path.exists(
            model_path_saved
        ), "The Saved Weights are not present"  # noqa

        # If the weights are present then can create samples directory if not present:  # noqa
        if not os.path.exists(sample_path):
            os.mkdir(sample_path)

        # Check that there are no samples in the sample path:
        assert not os.listdir(sample_path), "The Sample folder is not empty"

        # Check if user used Graphs when training:
        if args.use_graph_in_model:
            # Setup Graph Diffusion model:
            lit_diff_model = LitDiffusionModel_With_graph(
                in_node_nf=in_node_nf,
                in_edge_nf=2,
                hidden_features=args.hidden_features,
                n_layers=args.n_layers,
                device=device,
                lr=args.lr,
                test_sampling_number=args.test_sampling_number,
                save_samples=True,
                save_path=sample_path,
                timesteps=args.timesteps,
                noise_schedule=args.noise_schedule,
                learning_rate_schedule=args.learning_rate_scheduler,
                no_product=args.remove_product,
                batch_size=args.batch_size,
                pytest_time=pytest_time,  # ADDED HERE CHECK IT WORKS
            )

        else:
            # Setup regular model:
            lit_diff_model = LitDiffusionModel(
                dataset_to_use=args.dataset_to_use,
                in_node_nf=in_node_nf,
                hidden_features=args.hidden_features,
                n_layers=args.n_layers,
                device=device,
                lr=args.lr,
                remove_hydrogens=args.remove_hydrogens,
                test_sampling_number=args.test_sampling_number,
                save_samples=True,
                save_path=sample_path,
                timesteps=args.timesteps,
                noise_schedule=args.noise_schedule,
                random_rotations=args.random_rotations,
                augment_train_set=args.augment_train_set,
                include_context=args.include_context,
                learning_rate_schedule=args.learning_rate_scheduler,
                no_product=args.remove_product,
                batch_size=args.batch_size,
                pytest_time=pytest_time,  # ADDED HERE CHECK IT WORKS
            )

        # Load the State Dict:
        try:
            lit_diff_model.load_state_dict(torch.load(model_path_saved))
        except RuntimeError:
            print("\nWrong Weights for the model you are trying to load.")
            print(
                "Please check the Config File and the model you are trying",
                "to load.",
            )
            exit()

        # No Logger needed during Sampling:
        logger = None

        if pytest_time:
            # Cannot use GPU:
            # Setup trainer with CPU:
            trainer = pl.Trainer(
                accelerator="cpu",
                logger=logger,
                fast_dev_run=False,
            )
        else:
            # Setup trainer with GPU:
            trainer = pl.Trainer(
                accelerator="cuda",
                logger=logger,
                fast_dev_run=False,
            )
        trainer.test(lit_diff_model)

    else:
        raise (AssertionError, "Please use either train or test")


if __name__ == "__main__":
    # Setup the Argument Parser:
    parser = argparse.ArgumentParser(description="TransitionDiff")
    parser.add_argument(
        "--config",
        type=argparse.FileType(mode="r"),
        default="configs/train_diffusion.yml",
    )  # Read the Config File

    # Model Type Arguments:
    parser.add_argument(
        "--use_graph_in_model",
        type=bool,
        default=False,
    )  # If true then use Graph Model

    # Wether we are in training or testing mode:
    parser.add_argument(
        "--train_or_test", type=str, default="train"
    )  # Wether we are in train or test mode

    # Dataset Arguments:
    parser.add_argument(
        "--dataset_to_use", type=str, default="W93"
    )  # Number of diffusion steps

    # Diffusion Arguments:
    parser.add_argument(
        "--timesteps", type=int, default=2_000
    )  # Number of diffusion steps
    parser.add_argument(
        "--noise_schedule", type=str, default="sigmoid_2"
    )  # Different types of Noise Schedules available
    parser.add_argument(
        "--remove_hydrogens", type=bool, default=False
    )  # Wether or not to include hydrogens during training or not
    parser.add_argument(
        "--random_rotations", type=bool, default=False
    )  # Form of data-augmentation --> Randomly rotates all the coordinates in E(3) Fashion  # noqa
    parser.add_argument(
        "--augment_train_set", type=bool, default=True
    )  # Augment the train set by duplicating each train reation but replacing the product with the reactant and vice versa  # noqa
    parser.add_argument("--include_context", type=bool, default=False)
    parser.add_argument(
        "--remove_product",
        type=bool,
        default=False,
    )  # Do Not use Product during training

    # EGNN Arguments:
    parser.add_argument(
        "--n_layers", type=int, default=5
    )  # Number of messagepassing layers
    parser.add_argument(
        "--hidden_features", type=int, default=64
    )  # size of the projection of the conditional vector H between each message passing layer  # noqa

    # Training arguments:
    parser.add_argument(
        "--lr", type=float, default=1e-4
    )  # Leaning rate --> Have to be quite small as model has tendency to blow-up  # noqa
    parser.add_argument(
        "--epochs", type=int, default=100
    )  # Number of training Epochs  # noqa
    parser.add_argument(
        "--batch_size", type=int, default=64
    )  # Size of batchsize really depends on the dataset you are using --> the TS dataset is very small so could really scale this up  # noqa
    parser.add_argument(
        "--learning_rate_scheduler", type=bool, default=False
    )  # Wether or not to use a learning rate Schedule --> In our case it could be beneficial not too as training can be pretty stochastic  # noqa
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
    )  # Path to Model weights
    parser.add_argument(
        "--folder_name",
        type=str,
        default=None,
    )  # Path to Samples

    # May have to add WanB stuff here:
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="Diffusion_5_layer_2000_timesteps",
    )  # Controls the name of the project on WandB

    parser.add_argument(
        "--test_sampling_number", type=int, default=10
    )  # Number of samples to save during sampling time

    # Parse the command line arguments:
    args = parser.parse_args()

    # If there is a YAML file available:
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            arg_dict[key] = value
        args.config = args.config.name
    else:
        config_dict = {}
    main(args=args)
