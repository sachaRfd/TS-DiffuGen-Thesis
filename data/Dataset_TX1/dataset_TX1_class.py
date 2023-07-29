import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))



import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.data import Dataset, Data
from torch.utils.data.dataset import Subset
from torch_geometric.utils import erdos_renyi_graph
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Dataset_TX1.TX1_dataloader import Dataloader as TX1_Loader



class TX1_dataset(Dataset):
    def __init__(self, directory="Dataset_TX1/Transition1x.h5", split="data"):
        super().__init__()

        # Load TX1 dataloader: 
        self.dataloader = TX1_Loader(directory, split, only_final=True)
        self.data = []
        self.node_mask = []
        self.max_length = 23

        # OHE TO atomic number encoding: 
        self.ohe_to_atomic_number = {1: [0, 0, 0, 1],
                        6: [1, 0, 0, 0],
                        7: [0, 1, 0, 0],
                        8: [0, 0, 1, 0]}
        # For Sampling
        self.ohe_dict = {"C": [1, 0, 0, 0],
                        "N": [0, 1, 0, 0],
                        "O": [0, 0, 1, 0],
                        "H": [0, 0, 0, 1]}



        self.setup()
        assert len(self.data) == len(self.node_mask)
    
    def setup(self):
        """
        Setup the Data and the node masks
        """
        for molecule in self.dataloader:
            # Atom type is in atomic number, lets convert it back to a OHE representation: 
            atom_type = molecule["transition_state"]["atomic_numbers"]
            

            # Get the OHE Representations: 
            ohe = []
            for atom in atom_type: 
                ohe.append(self.ohe_to_atomic_number.get(atom))


            # Get the coordinates: 
            ts_coordinates = molecule["transition_state"]["positions"]
            r_coordinates = molecule["reactant"]["positions"]
            p_coordinates = molecule["product"]["positions"]
            
            # Now we can setup a molecule list that will be fed through a downstream sampling model:
            mol = []
            for ohe_atom, reactant, product, ts in zip(ohe, r_coordinates, p_coordinates, ts_coordinates):
                combined_values = [*ohe_atom, *reactant, *product, *ts]
                mol.append(combined_values)
                

            # Check that the mean of each variable is 0: 
            mol = torch.tensor(mol, dtype=torch.float32)

            # Remove the mean from the reactants, products and transition states:
            mol[:, 4:7] -= mol[:, 4:7].mean(axis=0)
            mol[:, 7:10] -= mol[:, 7:10].mean(axis=0)
            mol[:, 10:13] -= mol[:, 10:13].mean(axis=0)


            # Check that the mean of each variable is 0: 
            assert np.allclose(mol[:, 4:7].mean(axis=0), 0, atol=1e-5)
            assert np.allclose(mol[:, 7:10].mean(axis=0), 0, atol=1e-5)
            assert np.allclose(mol[:, 10:13].mean(axis=0), 0, atol=1e-5)

            # Create the padding:
            padding = torch.tensor([[0.0] * mol.shape[1]] * (self.max_length - mol.shape[0]), dtype=torch.float32)
            
            # Concatenate the padding to the original molecule tensor: 
            padded_mol = torch.concatenate((mol, padding), dim=0)
            # print(padded_mol.shape)
            # exit()

            # Now create the node mask: 
            node_mask = torch.tensor([1.0] * mol.shape[0] + [0.0] * (self.max_length - mol.shape[0]), dtype=torch.float32)
        
            # Append them to dataset: 
            self.node_mask.append(node_mask)
            self.data.append(padded_mol)
        
        print(f"\nTX1 Dataset has been loaded successfully.\nThe dataset contains {len(self.data)} molecules\n")




    

    def get(self, idx):
        return self.data[idx], self.node_mask[idx]
    
    def len(self):
        return len(self.data)



if __name__ == "__main__":
    # Split can be train, test, val, data
    dataset = TX1_dataset(split="data")
    print(f"Total Dataset size {dataset.len()}")

    # We want to match the sizes used in MIT paper 9,000 for training and 1,073 for testing
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.1, random_state=42, shuffle=True)

    # Split the training set into a 8:1 split to train/val set: 
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=1/9, random_state=42, shuffle=True)
    print(f"Train set size:\t{len(train_dataset)}\tVal set size:\t{len(val_dataset)}\tTest set size:\t{len(test_dataset)}")


    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)
    train_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)



    # print(dataset[2][0].shape, dataset[2][1].shape)
    # print(next(iter(dataloader)))



    # print(dataset[200])
