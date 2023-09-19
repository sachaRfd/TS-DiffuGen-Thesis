# Sacha Raffaud sachaRfd and acse-sr1022

import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from data.Dataset_TX1.TX1_dataloader import Dataloader as TX1_Loader


"""


This script contains the TX1 pytorch dataset class. 
We wanted it to inherit from the W93 dataset class, however, as the dataset files
are storred in a different format it requires to be loaded through a dataloader. 
Optimisations could be done but not that interesting.

- Takes a little while longer because have to iterate over the whole TX1_Loader to 
set it up in correct way.


When this file is called, the TX1 dataset is instanciated and the different sizes of 
each dataset split is returned.

Not really testable unless a .h5 file containing a small amount of samples is used.
I did not want to overwelm the Repo with that. 
"""  # noqa


class TX1_dataset(Dataset):
    def __init__(
        self,
        directory="data/Dataset_TX1/Transition1x.h5",
        split="data",
    ):
        super().__init__()

        # Load TX1 dataloader:
        self.dataloader = TX1_Loader(directory, split, only_final=True)
        self.data = []
        self.node_mask = []
        self.max_length = 23

        # OHE TO atomic number encoding:
        self.ohe_to_atomic_number = {
            1: [0, 0, 0, 1],
            6: [1, 0, 0, 0],
            7: [0, 1, 0, 0],
            8: [0, 0, 1, 0],
        }
        # For Sampling
        self.ohe_dict = {
            "C": [1, 0, 0, 0],
            "N": [0, 1, 0, 0],
            "O": [0, 0, 1, 0],
            "H": [0, 0, 0, 1],
        }

        self.setup()
        assert len(self.data) == len(self.node_mask)

    def setup(self):
        """
        Setup the Data and the node masks
        """
        for molecule in self.dataloader:
            # Atom type is in atomic number, lets convert it back to a OHE representation:  # noqa
            atom_type = molecule["transition_state"]["atomic_numbers"]

            # Get the OHE Representations:
            ohe = []
            for atom in atom_type:
                ohe.append(self.ohe_to_atomic_number.get(atom))

            # Get the coordinates:
            ts_coordinates = molecule["transition_state"]["positions"]
            r_coordinates = molecule["reactant"]["positions"]
            p_coordinates = molecule["product"]["positions"]

            # Now we can setup a molecule list that will be fed through a downstream sampling model: # noqa
            mol = []
            for ohe_atom, reactant, product, ts in zip(
                ohe, r_coordinates, p_coordinates, ts_coordinates
            ):
                combined_values = [*ohe_atom, *reactant, *product, *ts]
                mol.append(combined_values)

            # Check that the mean of each variable is 0:
            mol = torch.tensor(mol, dtype=torch.float32)

            # Remove the mean from the reactants, products and transition states: # noqa
            mol[:, 4:7] -= mol[:, 4:7].mean(axis=0)
            mol[:, 7:10] -= mol[:, 7:10].mean(axis=0)
            mol[:, 10:13] -= mol[:, 10:13].mean(axis=0)

            # Check that the mean of each variable is 0:
            assert np.allclose(mol[:, 4:7].mean(axis=0), 0, atol=1e-5)
            assert np.allclose(mol[:, 7:10].mean(axis=0), 0, atol=1e-5)
            assert np.allclose(mol[:, 10:13].mean(axis=0), 0, atol=1e-5)

            # Create the padding:
            padding = torch.tensor(
                [[0.0] * mol.shape[1]] * (self.max_length - mol.shape[0]),
                dtype=torch.float32,
            )

            # Concatenate the padding to the original molecule tensor:
            padded_mol = torch.concatenate((mol, padding), dim=0)

            # Now create the node mask:
            node_mask = torch.tensor(
                [1.0] * mol.shape[0] + [0.0] * (self.max_length - mol.shape[0]),  # noqa
                dtype=torch.float32,
            )

            # Append them to dataset:
            self.node_mask.append(node_mask)
            self.data.append(padded_mol)

        print(
            f"\nTX1 Dataset has been loaded successfully.\nThe dataset contains {len(self.data)} molecules\n"  # noqa
        )

    def get(self, idx):
        return self.data[idx], self.node_mask[idx]

    def len(self):
        return len(self.data)


if __name__ == "__main__":
    # Split can be train, test, val, data
    dataset = TX1_dataset(split="data")
    print(f"Total Dataset size {dataset.len()}")

    # We want to match the sizes used in MIT paper 9,000 for training and 1,073 for testing # noqa
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.1, random_state=42, shuffle=True
    )

    # Split the training set into a 8:1 split to train/val set:
    train_dataset, val_dataset = train_test_split(
        train_dataset, test_size=1 / 9, random_state=42, shuffle=True
    )
    print(
        f"Train set size:\t{len(train_dataset)}\tVal set size:\t{len(val_dataset)}\tTest set size:\t{len(test_dataset)}"  # noqa
    )

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=64, shuffle=True
    )  # noqa
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=64, shuffle=False
    )  # noqa
    train_dataloader = DataLoader(
        dataset=test_dataset, batch_size=64, shuffle=False
    )  # noqa

    print(next(iter(train_dataloader)))
