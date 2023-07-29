import os
import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.data import Dataset, Data
from torch.utils.data.dataset import Subset
from torch_geometric.utils import erdos_renyi_graph
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



"""
This dataset class should only include the reactants

1. So OHE of atom types, then reactants, then TS

"""


class QM90_TS_no_reactants(Dataset):

    def __init__(self, directory="Diffusion_Project/Dataset/data/Clean_Geometries",
                 remove_hydrogens=False,
                 graph=False,
                 plot_distribution=False, 
                 include_context=False):
        super().__init__()
        # First we can read all the data from the files:
        self.remove_hydrogen = remove_hydrogens
        self.directory = directory
        self.context = include_context
        self.count = 0
        
        # Dictionaries
        self.atom_dict = {}
        self.ohe_dict = {}
        self.nuclear_charges = {"H":1,"C":6, "N":7,"O":8, "None":0} # Dictionary containing the nuclear charges of the atoms
        self.van_der_waals_radius = {"H":1.20,"C":1.70, "N":1.55,"O":1.52, "None":0} # Dictionary containing the van_der_waals_radius of the atoms in Argstroms

        self.data = []
        self.reactant = []
        # self.product = []
        self.transition_states = []

        # Node masks: 
        self.node_mask = []


        # Run Assert for path:        
        assert os.path.exists(self.directory)


        # Run the Setup: 
        self.count_data() 
              
        # Append the reaction data to tbe data-variale
        for reaction_number in range(self.count):
            self.extract_data(reaction_number)
        

        # Count the atoms: 
        self.atom_count()

        # Print the atom count: 
        print(f"\nThe dataset includes {self.count} reactions and the following atom count:\n")
        for atom, count in self.atom_dict.items():
            print(f"\t{atom}: {count}")
        print()


        if plot_distribution:
            # Plot the size distribution of the molecules - Before we One hot encode: 
            self.plot_molecule_size_distribution()

        # One Hot Encode the atoms: 
        self.one_hot_encode()
  
        # Assert that the shapes are correct:
        assert self.reactant.shape == self.transition_states.shape

        if graph:
            # Create the graphs: 
            print("Creating Graphs")
            self.create_graph()
        else:
            print("Not Using graphs")
            self.create_data_array()

        print("\nFinished creating the dataset. ")

    def count_data(self):
        # See how many sub-folders are present: 
        for folder in os.listdir(self.directory):
            if folder.startswith("Reaction_"):
                self.count += 1
  
    
    def extract_data(self, reaction_number):
        # Get the Full path: 
        path = os.path.join(self.directory, f"Reaction_{reaction_number}")
        assert os.path.exists(path)  # Assert the path Exists

        # Check that in the path there are the three files:
        assert len(os.listdir(path)) == 4, "The folder is missing files."  # 4 FIles as we have the reactant images also in the directory

        # Now we can extract the Reactant, Product, TS info:
        for file in os.listdir(path):#
            if file.startswith("Reactant"):
                # Append to Reactant Matrix:
                reactant_matrix = []
                with open(os.path.join(path, file), "r") as read_file:
                    lines = read_file.readlines()
                    for line in lines:
                        if self.remove_hydrogen:
                            # Check that the line is not Oxygens: 
                            if line[0] != "H":
                                reactant_matrix.append(line.split())
                        else: 
                            reactant_matrix.append(line.split())
                self.reactant.append(reactant_matrix)

            # elif file.startswith("Product"):
            #     # Append to Reactant Matrix:
            #     product_matrix = []
            #     with open(os.path.join(path, file), "r") as read_file:
            #         lines = read_file.readlines()
            #         for line in lines:
            #             if self.remove_hydrogen:
            #                 if line[0] != "H":
            #                     product_matrix.append(line.split())
            #             else:     
            #                 product_matrix.append(line.split())
            #     # self.product.append(product_matrix)

            elif file.startswith("TS"):
                # Append to Reactant Matrix:
                ts_matrix = []
                with open(os.path.join(path, file), "r") as read_file:
                    lines = read_file.readlines()
                    for line in lines: 
                        if self.remove_hydrogen:
                            if line[0] != "H":
                                ts_matrix.append(line.split())
                        else:
                            ts_matrix.append(line.split())
                self.transition_states.append(ts_matrix)


    def atom_count(self):
        # Iterate over all the values in the lists to find if the different molecules: 
        for mol in self.reactant:
            for atom in mol: 
                if atom[0] not in self.atom_dict:
                    self.atom_dict[atom[0]] = 1
                else:
                    self.atom_dict[atom[0]] +=1



    def plot_molecule_size_distribution(self):
        # Count the size of each molecule
        molecule_sizes = [len(mol) for mol in self.reactant + self.product + self.transition_states]

        # Create a dictionary to store the count of each molecule size
        molecule_size_count = {}
        for size in molecule_sizes:
            if size not in molecule_size_count:
                molecule_size_count[size] = 1
            else:
                molecule_size_count[size] += 1

        # Sort the molecule sizes in ascending order
        sorted_sizes = sorted(molecule_size_count.keys())

        # Create lists for x-axis (molecule sizes) and y-axis (count)
        x = [str(size) for size in sorted_sizes]
        y = [molecule_size_count[size] for size in sorted_sizes]

        # Plot the bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(x, y)
        plt.xlabel('Molecule Size', fontsize=15)  # Increase font size for x-axis label
        plt.ylabel('Count', fontsize=15)  # Increase font size for y-axis label
        plt.title('Distribution of Molecule Sizes', fontsize=16)  # Increase font size for title
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def one_hot_encode(self):
        num_of_atoms = len(self.atom_dict)

        for index, atom in enumerate(self.atom_dict):
            ohe_vector = [0] * num_of_atoms
            ohe_vector[index] = 1  # Set the position corresponding to the atom index to 1
            self.ohe_dict[atom] = ohe_vector
        
        print(f"\nThe Atom Encoding is the following:\n")
        for atom, count in self.ohe_dict.items():
            print(f"\t{atom}: {count}")
            # print(type(count))
        print()

        # Replace the atom str with OHE vector
        self.replace_atom_types_with_ohe_vectors(self.reactant)
        # self.replace_atom_types_with_ohe_vectors(self.product)
        self.replace_atom_types_with_ohe_vectors(self.transition_states)

        # Convert everything in the self.reactants, self.products, and self.transition_states to floats:
        self.reactant = [[[float(value) for value in atom] for atom in mol] for mol in self.reactant]
        # self.product = [[[float(value) for value in atom] for atom in mol] for mol in self.product]
        self.transition_states = [[[float(value) for value in atom] for atom in mol] for mol in self.transition_states]

        # Calculate the center of gravity and remove it
        self.delete_centre_gravity()  # This has been tested in the testing images on local computer --> WORKS


        # Convert everything in the self.reactant, self.product, and self.transition_states back to lists
        self.reactant = [mol.tolist() for mol in self.reactant]
        # self.product = [mol.tolist() for mol in self.product]
        self.transition_states = [mol.tolist() for mol in self.transition_states]


        # Check the maximum length among all nested lists for padding
        # max_length = max(len(mol) for mol in self.reactant + self.product + self.transition_states)
        max_length = max(len(mol) for mol in self.reactant + self.transition_states)



        # Pad the nested lists to have the same length
        padded_reactant = [mol + [[0.0] * len(mol[0])] * (max_length - len(mol)) for mol in self.reactant]
        # padded_product = [mol + [[0.0] * len(mol[0])] * (max_length - len(mol)) for mol in self.product]
        padded_transition_states = [mol + [[0.0] * len(mol[0])] * (max_length - len(mol)) for mol in self.transition_states]

        # Make a copy of the unpadded chemicals
        self.unpadded_chemical = self.reactant.copy()

        # Create the node mask:
        self.node_mask = torch.tensor([[1.0] * len(mol) + [0.0] * (max_length - len(mol)) for mol in self.reactant])


        # Assign the padded values to self.reactant, self.product, and self.transition_states
        self.reactant = torch.tensor(padded_reactant)
        # self.product = torch.tensor(padded_product)
        self.transition_states = torch.tensor(padded_transition_states)
    
    
    
    def replace_atom_types_with_ohe_vectors(self, molecule_list):
        for mol in molecule_list:
            for atom in mol:
                atom_type = atom[0]  # Get the atom type
                if atom_type in self.ohe_dict:
                    ohe_vector = self.ohe_dict[atom_type]
                    atom[0:1] = ohe_vector
    
    
    def delete_centre_gravity(self):
        # Calculate the center of gravity for each molecule
        for index in range(self.count):
            reactant_coords = np.array(self.reactant[index])[:, len(self.atom_dict):].astype(float)
            # product_coords = np.array(self.product[index])[:, len(self.atom_dict):].astype(float)
            ts_coords = np.array(self.transition_states[index])[:, len(self.atom_dict):].astype(float)

            # Calculate the center of gravity
            reactant_center = np.mean(reactant_coords, axis=0)
            # product_center = np.mean(product_coords, axis=0)
            ts_center = np.mean(ts_coords, axis=0)

            # Remove the center of gravity from each molecule
            self.reactant[index] = np.array(self.reactant[index])
            self.reactant[index][:, len(self.atom_dict):] = reactant_coords - reactant_center

            # self.product[index] = np.array(self.product[index])
            # self.product[index][:, len(self.atom_dict):] = product_coords - product_center

            self.transition_states[index] = np.array(self.transition_states[index])
            self.transition_states[index][:, len(self.atom_dict):] = ts_coords - ts_center

    def create_graph(self):
        # We can append it to the self.data tensor of graphs, and make the positions the TS, we also make it fully connected
        for index in range(self.count):
            # Create Fully connected graph
            num_nodes = self.reactant[index].shape[0]
            edge_index = erdos_renyi_graph(num_nodes,   edge_prob=1.0)
            
            graph = Data(
                x=torch.cat([self.reactant[index, :, :], self.product[index, :, -3:], self.transition_states[index, :, -3:]], dim=1),
                pos=self.transition_states[index, :, -3:],
                edge_index=edge_index
            )
            self.data.append(graph)
    
    def get_keys_from_value(self, search_value):
        result = [key for key, value in self.ohe_dict.items() if value == search_value]
        return result if result else ["None"]   # Make it return none if it comes upon the masked atoms
    
    def create_data_array(self):
        # Check if we want to add context to the data: 
        if self.context: 
            # Add this to utils file later
            print("Adding the context to the dataset")
            print("Including nuclear charges")
            # This is to just return data as simple matrix and not graph
            for index in range(self.count):
                # Let's get the nuclear charge of the atom from the OHE: 
                ohes = self.reactant[index, :, :-3]


                # Get the atomic features: 
                atomic_feature_vector = ohes

                # Get the Van Der Waals RADII of all the atoms: 
                atom_types = [self.get_keys_from_value(atom.tolist()) for atom in atomic_feature_vector]    # Get the atom type
                van_der_waals = torch.tensor([[self.van_der_waals_radius[atom_type[0]]] for atom_type in atom_types])  # Get the nuclear charge from the atom type


                # concatenate the OHE, the Nuclear charge, and then the coordinates of the reactant/product/TS               
                x=torch.cat([ohes, van_der_waals, self.reactant[index, :, -3:], self.product[index, :, -3:], self.transition_states[index, :, -3:]], dim=1)
                self.data.append(x)

        else: 
            print("Not including Context information")

            # This is to just return data as simple matrix and not graph
            for index in range(self.count):
                # Only take the last 3 parts of the product and transition states as they all contain the OHE
                # x=torch.cat([self.reactant[index, :, :], self.product[index, :, -3:], self.transition_states[index, :, -3:]], dim=1)
                x=torch.cat([self.reactant[index, :, :], self.transition_states[index, :, -3:]], dim=1)

                self.data.append(x)


    
    def get(self, idx):
        # Remove the mean --> So that we have translation invariant data
        return self.data[idx], self.node_mask[idx]


    def len(self):
        return len(self.data)



if __name__ == "__main__":
    remove_hydrogens = False
    dataset = QM90_TS_no_reactants(directory="Dataset/data/Clean_Geometries/",
                      remove_hydrogens=remove_hydrogens,
                      graph=False,
                      plot_distribution=False,
                      include_context=True)


    # In all papers they use 8:1:1 ratio
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=42)

    batch_size = 20
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    print(next(iter(train_loader))[0][0][0])








