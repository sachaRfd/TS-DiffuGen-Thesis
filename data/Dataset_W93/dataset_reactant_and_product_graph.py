# import numpy as np
import pandas as pd

# from rdkit import Chem
# import os

# import torch
# from torch_geometric.data import Data
# from torch.utils.data import DataLoader

"""     # noqa

Script to read the samples and create fully connected graphs
which capture the whole reaction information 
    - Reactant graph and coordinates
    - Product graph

    

Node Feature includes: 
- OHE of atom, XYZ coordinates of reactant and XYZ Coordinates of TS

Edge Feature:
- Fully connected graph with bond type in reactant and product as edge features

"""


if __name__ == "__main__":
    data = pd.read_csv(
        "data/Dataset_W93/data/w93_dataset/wb97xd3.csv", index_col=0
    )  # noqa

    reactant = data.iloc[1].rsmi
    print(reactant)
