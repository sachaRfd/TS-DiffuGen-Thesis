import argparse
import yaml
import os

import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor


from src.train import LitDiffusionModel
from src.EGNN import dynamics


def main(args):
    # Setup the Model Names and the paths automatically
    if args.automatic_model_name:
        model_name = f"{args.include_context}_include_VAN_DER_WAAL_RADII_{args.random_rotations}_Random_rotations_{args.augment_train_set}_augment_train_set_{args.n_layers}_layers_{args.hidden_features}_hiddenfeatures_{args.lr}_lr_{args.noise_schedule}_{args.timesteps}_timesteps_{args.batch_size}_batch_size_{args.epochs}_epochs_{args.remove_hydrogens}_Rem_Hydrogens"  # noqa
        folder_name = "src/Diffusion/Clean_lightning/" + model_name + "/"
    else:
        model_name = args.model_save_path
        folder_name = args.folder_name

    # Add weights and sample path to folder path:
    model_path = folder_name + "Weights/"
    sample_path = folder_name + "Samples/"
    model_path_saved = os.path.join(model_path, "weights.pth")

    # if not os.path.exists(folder_name):
    #     os.makedirs(folder_name)

    # if not os.path.exists(model_path):
    #     os.makedirs(model_path)

    # if not os.path.exists(sample_path):
    #     os.makedirs(sample_path)

    # Create the directories if we are in training mode:
    if args.train_test == "train":
        # Make sure the folders are present else create them:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if not os.path.exists(sample_path):
            os.makedirs(sample_path)

    # Calculate the In Node number depending on hydrogen conditions and if there is context present in node features:  # noqa
    if args.remove_hydrogens:
        in_node_nf = 9 + 1  # To account for time and 1 less OHE
    else:
        in_node_nf = 10 + 1  # To account for time

    if args.include_context:
        in_node_nf += 1  # Add one for the size of context --> For now we just have the Nuclear Charge  # noqa

    # Setup the GPU: PUT THIS IN THE UTILS FILE INSTEAD OF MODEL DYNAMICS WITH MASK  # noqa
    device = dynamics.setup_device()

    # Setup the model:
    lit_diff_model = LitDiffusionModel(
        dataset_to_use=args.dataset_to_use,
        in_node_nf=in_node_nf,
        context_nf=args.context_nf,
        hidden_features=args.hidden_features,
        out_node=args.out_node,
        n_dims=args.n_dims,
        n_layers=args.n_layers,
        sin_embedding=True,
        device=device,
        lr=args.lr,
        remove_hydrogens=args.remove_hydrogens,
        test_sampling_number=args.test_sampling_number,
        save_samples=args.save_samples,
        save_path=sample_path,
        timesteps=args.timesteps,
        noise_schedule=args.noise_schedule,
        loss_type=args.loss_type,
        random_rotations=args.random_rotations,
        augment_train_set=args.augment_train_set,
        include_context=args.include_context,
        learning_rate_schedule=args.learning_rate_scheduler,
    )

    # Setup the trainer if we are in train or test set:
    if args.train_test == "train":
        # train mode:
        wandb_logger = pl.loggers.WandbLogger(
            project=args.wandb_project_name, name=model_name
        )

        # Learning rate scheduler:
        if args.learning_rate_scheduler:
            lr_monitor = LearningRateMonitor(logging_interval="epoch")
            trainer = pl.Trainer(
                accelerator="cuda",
                max_epochs=args.epochs,
                logger=wandb_logger,
                callbacks=[lr_monitor],
                fast_dev_run=False,
            )

            trainer.fit(lit_diff_model)

        else:
            lr_monitor = LearningRateMonitor(logging_interval="epoch")
            trainer = pl.Trainer(
                accelerator="cuda",
                max_epochs=args.epochs,
                logger=wandb_logger,
                fast_dev_run=False,
            )  # Fast develop run was used for debugging purposes

            trainer.fit(lit_diff_model)

        # Save the model:
        torch.save(lit_diff_model.state_dict(), model_path_saved)
        wandb.finish()

    elif args.train_test == "test":
        # Test mode: So sample the test set:
        # Load the State Dict:
        lit_diff_model.load_state_dict(torch.load(model_path_saved))

        # No need for Logger when testing currently:
        logger = None

        trainer = pl.Trainer(
            accelerator="cuda", logger=logger, fast_dev_run=False
        )  # noqa
        trainer.test(lit_diff_model)  # noqa


if __name__ == "__main__":
    # Setup the Argument Parser:
    parser = argparse.ArgumentParser(description="TransitionDiff")
    parser.add_argument(
        "--config",
        type=argparse.FileType(mode="r"),
        default="configs/simple_diff_train.yml",
    )  # Read the Config File

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
        "--loss_type", type=str, default="l2"
    )  # Remove this from whole setup
    parser.add_argument(
        "--remove_hydrogens", type=bool, default=False
    )  # Wether or not to include hydrogens during training or not
    parser.add_argument(
        "--random_rotations", type=bool, default=False
    )  # Form of data-augmentation --> Randomly rotates all the coordinates in E(3) Fashion  # noqa
    parser.add_argument(
        "--augment_train_set", type=bool, default=True
    )  # Augment the train set by duplicating each train reation but replacing the product with the reactant and vice versa  # noqa
    parser.add_argument(
        "--include_context", type=bool, default=False
    )  # IMPLEMENATION IS NOT 100% yet

    # Wether we are in training or testing mode:
    parser.add_argument(
        "--train_test", type=str, default="train"
    )  # Wether we are in train or test mode

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
        "--automatic_model_name", type=bool, default=True
    )  # Wether or not to just name the model after all the parameters used
    parser.add_argument(
        "--model_save_path", type=str, default=None
    )  # For now we just them automatically
    parser.add_argument(
        "--folder_name", type=str, default=None
    )  # For now we just do it automatically

    # May have to add WanB stuff here:
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="Diffusion_5_layer_2000_timesteps",  # noqa
    )  # Controls the name of the project on WandB
    # Will have to add stuff on loging in here and stuff --> Look into this later on  # noqa
    # May have to implement simple tensorboard stuff

    # EGNN Arguments:
    parser.add_argument(
        "--out_node", type=int, default=3
    )  # Output dimension --> we want XYZ coordinates so 3
    parser.add_argument(
        "--context_nf", type=int, default=0
    )  # Global context size --> None for now
    parser.add_argument(
        "--n_dims", type=int, default=3
    )  # X dimension --> 3 for the XYZ coordinates
    parser.add_argument(
        "--n_layers", type=int, default=5
    )  # Number of messagepassing layers
    parser.add_argument(
        "--hidden_features", type=int, default=64
    )  # size of the projection of the conditional vector H between each message passing layer  # noqa

    # Testing Arguments:
    parser.add_argument(
        "--save_samples", type=bool, default=True
    )  # Wether or not to save the samples during test time
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
            # if isinstance(value, list) and key != 'normalize_factors':
            #     for v in value:
            #         arg_dict[key].append(v)
            # else:
            arg_dict[key] = value
        args.config = args.config.name
    else:
        config_dict = {}
    main(args=args)
