# import numpy as np
import os
import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

""" 
# noqa

Script to read the samples and create fully connected graphs
which capture the whole reaction information 
    - Reactant graph and coordinates
    - Product graph

    

Node Feature includes: 
- OHE of atom, XYZ coordinates of reactant and XYZ Coordinates of TS

Edge Feature:
- Fully connected graph with bond type in reactant and product as edge features

"""


class W93_TS_coords_and_reacion_graph(Dataset):
    def __init__(
        self,
        directory="data/Dataset_W93/data/Clean_Geometries/",
        running_pytest=False,
    ):
        super().__init__()

        # First we can read all the data from the files:
        self.directory = directory
        # Run Assert for path:
        assert os.path.exists(self.directory)
        self.count = 0
        self.running_pytest = running_pytest

        # Dictionaries
        self.atom_dict = {}
        self.ohe_dict = {}

        self.data = []
        self.reactant = []
        self.product = []
        self.transition_states = []

        # List for edge_attributes:
        self.edge_attributes = []

        # Node masks:
        self.node_mask = []

        self.load_data()
        print("Finished creating the dataset.")

    def load_data(self):
        """# noqa
        Loads the dataset by extracting data from reaction files and performing preprocessing steps.
        """
        self.count_data()
        for reaction_number in range(self.count):
            self.extract_data(reaction_number)
        self.atom_count()

        self.one_hot_encode()

        assert (
            self.reactant.shape
            == self.product.shape
            == self.transition_states.shape  # noqa
        )  # noqa

        self.create_data_array()

        # integrate the bond information as edge features and return them
        self.setup_edge_attributes()

        # # Run the Setup:
        # self.count_data()

        # # Append the reaction data to tbe data-variale
        # for reaction_number in range(self.count):
        #     self.extract_data(reaction_number)

        # # Count the atoms:
        # self.atom_count()

        # # Print the atom count:
        # print(
        #     f"\nThe dataset includes {self.count} reactions and the following atom count:\n"  # noqa
        # )
        # for atom, count in self.atom_dict.items():
        #     print(f"\t{atom}: {count}")
        # print()

        # # One Hot Encode the atoms:
        # self.one_hot_encode()

        # # Assert that the shapes are correct:
        # assert (
        #     self.reactant.shape
        #     == self.product.shape
        #     == self.transition_states.shape  # noqa
        # )  # noqa

        # print("Reading XYZ Files")
        # self.create_data_array()

        # # integrate the bond information as edge features and return them
        # self.setup_edge_attributes()

        # print("\nFinished creating the dataset. ")

    def count_data(self):
        """
        Counts the number of reactions in the dataset.
        """
        # See how many sub-folders are present:
        for folder in os.listdir(self.directory):
            if folder.startswith("Reaction_"):
                self.count += 1

    def read_xyz_file(self, file_path):
        """
        Reads an XYZ file and returns its data.
        """
        data = []
        with open(file_path, "r") as read_file:
            lines = read_file.readlines()
            for line in lines[2:]:
                data.append(line.split())
        return data

    def extract_data(self, reaction_number):
        """# noqa
        Extracts reactant, product, and transition state information from specified reaction.
        """
        # Get the Full path:
        path = os.path.join(self.directory, f"Reaction_{reaction_number}")
        assert os.path.exists(path)  # Assert the path Exists

        # Check that in the path there are the three files:
        assert (
            len(os.listdir(path)) == 4
        ), "The folder is missing files."  # 4 FIles as we have the reactant images also in the directory  # noqa

        # Now we can extract the Reactant, Product, TS info:
        for file in os.listdir(path):
            if file.startswith("Reactant"):
                # Extract reactant matrix
                reactant_matrix = self.read_xyz_file(os.path.join(path, file))
                self.reactant.append(reactant_matrix)
            elif file.startswith("Product"):
                # Extract product matrix
                product_matrix = self.read_xyz_file(os.path.join(path, file))
                self.product.append(product_matrix)
            elif file.startswith("TS"):
                # Extract transition state matrix
                ts_matrix = self.read_xyz_file(os.path.join(path, file))
                self.transition_states.append(ts_matrix)

    # def extract_data(self, reaction_number):
    #     # Get the Full path:
    #     path = os.path.join(self.directory, f"Reaction_{reaction_number}")
    #     assert os.path.exists(path)  # Assert the path Exists

    #     # Check that in the path there are the three files:
    #     assert (
    #         len(os.listdir(path)) == 4
    #     ), "The folder is missing files."  # 4 FIles as we have the reactant images also in the directory   # noqa

    #     # Now we can extract the Reactant, Product, TS info:
    #     for file in os.listdir(path):  #
    #         if file.startswith("Reactant"):
    #             # Append to Reactant Matrix:
    #             reactant_matrix = []
    #             with open(os.path.join(path, file), "r") as read_file:
    #                 lines = read_file.readlines()
    #                 for line in lines[2:]:
    #                     reactant_matrix.append(line.split())
    #             self.reactant.append(reactant_matrix)

    #         elif file.startswith("Product"):
    #             # Append to Reactant Matrix:
    #             product_matrix = []
    #             with open(os.path.join(path, file), "r") as read_file:
    #                 lines = read_file.readlines()
    #                 for line in lines[2:]:
    #                     product_matrix.append(line.split())
    #             self.product.append(product_matrix)

    #         elif file.startswith("TS"):
    #             # Append to Reactant Matrix:
    #             ts_matrix = []
    #             with open(os.path.join(path, file), "r") as read_file:
    #                 lines = read_file.readlines()
    #                 for line in lines[2:]:
    #                     ts_matrix.append(line.split())
    #             self.transition_states.append(ts_matrix)

    def atom_count(self):
        """
        Counts the occurrences of different atom types in the reactions.
        """
        # Iterate over all the values in the lists to find if the different molecules:# noqa
        for mol in self.reactant:
            for atom in mol:
                if atom[0] not in self.atom_dict:
                    self.atom_dict[atom[0]] = 1
                else:
                    self.atom_dict[atom[0]] += 1

    def generate_one_hot_encodings(self, num_of_atoms):
        """Generates one-hot encodings for atom types."""
        for index, atom in enumerate(self.atom_dict):
            ohe_vector = [0] * num_of_atoms
            ohe_vector[index] = 1
            self.ohe_dict[atom] = ohe_vector

    def convert_to_float(self, data):
        """
        Convert nested list elements to floats.
        """
        data = [
            [[float(value) for value in atom] for atom in mol] for mol in data
        ]  # noqa
        return data

    def convert_to_list(self, data):
        """
        Convert nested list elements to Lists.
        """
        data = [mol.tolist() for mol in data]
        return data

    def pad_data(self, data, max_length):
        """]
        Pads molecule so that all have the same size
            and can be fed through batch_loader
        """
        data = [
            mol + [[0.0] * len(mol[0])] * (max_length - len(mol))
            for mol in data  # noqa
        ]  # noqa
        return torch.tensor(data)

    def one_hot_encode(self):
        """# noqa
        Performs one-hot encoding of atom types and prepares other data-processing.
        """
        num_of_atoms = len(self.atom_dict)

        # Generate OHE:
        self.generate_one_hot_encodings(num_of_atoms)

        print("\nThe Atom Encoding is the following:\n")
        for atom, count in self.ohe_dict.items():
            print(f"\t{atom}: {count}")
        print()

        # Replace the atom str with OHE vector
        self.replace_atom_types_with_ohe_vectors(self.reactant)
        self.replace_atom_types_with_ohe_vectors(self.product)
        self.replace_atom_types_with_ohe_vectors(self.transition_states)

        # Convert everything in the self.reactants, self.products, and self.transition_states to floats:  # noqa
        self.reactant = self.convert_to_float(self.reactant)
        self.product = self.convert_to_float(self.product)
        self.transition_states = self.convert_to_float(self.transition_states)

        # Calculate the center of gravity and remove it
        self.delete_centre_gravity()

        # Convert everything in the self.reactant, self.product, and self.transition_states back to lists  # noqa
        self.reactant = self.convert_to_list(self.reactant)
        self.product = self.convert_to_list(self.product)
        self.transition_states = self.convert_to_list(self.transition_states)

        # Check the maximum length among all nested lists for padding
        max_length = max(len(mol) for mol in self.reactant)

        # Pad the nested lists to have the same length
        self.reactant = self.pad_data(self.reactant, max_length=max_length)
        self.product = self.pad_data(self.product, max_length=max_length)
        self.transition_states = self.pad_data(
            self.transition_states, max_length=max_length
        )

        # Create the node mask:
        self.node_mask = torch.tensor(
            [
                [1.0] * len(mol) + [0.0] * (max_length - len(mol))
                for mol in self.reactant
            ]
        )

    def replace_atom_types_with_ohe_vectors(self, molecule_list):
        """# noqa
        Replaces atom types in molecule data with their one-hot encoded vectors.
        """
        for mol in molecule_list:
            for atom in mol:
                atom_type = atom[0]  # Get the atom type
                if atom_type in self.ohe_dict:
                    ohe_vector = self.ohe_dict[atom_type]
                    atom[0:1] = ohe_vector

    def delete_centre_gravity(self):
        """
        Removes the center of gravity from molecule coordinates.
        Needed to keep model roto-translation invariant
        """
        # Calculate the center of gravity for each molecule
        for index in range(self.count):
            reactant_coords = np.array(self.reactant[index])[
                :, len(self.atom_dict) :  # noqa
            ].astype(float)
            product_coords = np.array(self.product[index])[
                :, len(self.atom_dict) :  # noqa
            ].astype(float)
            ts_coords = np.array(self.transition_states[index])[
                :, len(self.atom_dict) :  # noqa
            ].astype(float)

            # Calculate the center of gravity
            reactant_center = np.mean(reactant_coords, axis=0)
            product_center = np.mean(product_coords, axis=0)
            ts_center = np.mean(ts_coords, axis=0)

            # Remove the center of gravity from each molecule
            self.reactant[index] = np.array(self.reactant[index])
            self.reactant[index][:, len(self.atom_dict) :] = (  # noqa
                reactant_coords - reactant_center
            )

            self.product[index] = np.array(self.product[index])
            self.product[index][:, len(self.atom_dict) :] = (  # noqa
                product_coords - product_center
            )

            self.transition_states[index] = np.array(
                self.transition_states[index]
            )  # noqa
            self.transition_states[index][:, len(self.atom_dict) :] = (  # noqa
                ts_coords - ts_center
            )

    def get_keys_from_value(self, search_value):
        """
        Retrieves atom types based on their one-hot encoded vectors.
        """
        result = [
            key for key, value in self.ohe_dict.items() if value == search_value  # noqa
        ]  # noqa
        return (
            result if result else ["None"]
        )  # Make it return none if it comes upon the masked atoms

    def create_data_array(self):
        """# noqa
        Creates data arrays.
        """
        # This is to just return data as simple matrix and not graph
        for index in range(self.count):
            # Only take the last 3 parts of the product and transition states as they all contain the OHE# noqa
            x = torch.cat(
                [
                    self.reactant[index, :, :],
                    self.product[index, :, -3:],
                    self.transition_states[index, :, -3:],
                ],
                dim=1,
            )
            self.data.append(x)

    def setup_edge_attributes(
        self,
    ):
        print(
            "Setting up edge attributes to contain bond information about reactant and product| May take some time."  # noqa
        )
        # Get the adjacency matrix for the fully connected graph:
        if self.running_pytest:
            edge_index = get_adj_matrix_no_batch(n_nodes=16)  # Smaller dataset
        else:
            edge_index = get_adj_matrix_no_batch(n_nodes=23)  # Smaller dataset

        for molecule in tqdm(self.data):
            h = molecule[:, :-3]
            reactant_bonds, product_bonds = bond_to_edge(
                h=h, edge_index=edge_index
            )  # noqa
            edge_info = torch.concatenate(
                [reactant_bonds, product_bonds], dim=1
            )  # noqa

            # append to edge data
            self.edge_attributes.append(edge_info)

        print("Finished setting up edge attributes.")

    def get(self, idx):
        """
        Retrieves a data sample and its corresponding node mask.
        """
        return (
            self.data[idx],
            self.node_mask[idx],
            self.edge_attributes[idx],
        )

    def len(self):
        """
        Returns the length of the dataset.
        """
        return len(self.data)


_edges_dict = {}


def get_adj_matrix_no_batch(n_nodes):
    """# noqa
    Get the adjacency matrix for a fully connected graph with the specified number of nodes.

    This function returns the adjacency matrix as edge indices (rows and columns) for a graph
    with the given number of nodes. If the adjacency matrix for the specified number of nodes
    has already been generated, it is retrieved; otherwise, the function recursively generates
    the adjacency matrix.

    Args:
        n_nodes (int): Number of nodes in the graph.

    Returns:
        list of torch.LongTensor: A list containing two LongTensors representing the rows
                                  and columns of the edge indices of the adjacency matrix.
    """
    if n_nodes in _edges_dict:
        # get edges for a single sample
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                rows.append(i)
                cols.append(j)
        edges = [
            torch.LongTensor(rows),
            torch.LongTensor(cols),
        ]
        return edges
    else:
        _edges_dict[n_nodes] = {}
        return get_adj_matrix_no_batch(n_nodes)


# Extra functions for now which let us get the bond information out from reactant/product coordinates# noqa
def bond_to_edge(h, edge_index):
    """# noqa
    Generate bond information between atoms in a reactant and product configuration.

    This function takes atom information and edge indices to create bond information
    for a reaction graph based on reactant and product configurations.

    Args:
        h (torch.Tensor): Tensor containing atom and configuration information.
                          Shape: [num_atoms, num_features]
        edge_index (torch.Tensor): Tensor containing edge indices of the graph.
                                  Shape: [2, num_edges]

    Returns:
        torch.Tensor, torch.Tensor: Tensors containing bond information for reactant and product edges.
                                    Each element represents a bond type: 1 for a bond present,
                                    and 0 for no bond present.

    Note:
        - The input tensor h contains atom information and configuration details.
          It is structured as follows:
            - h[:, :4] contains atom type encodings.
            - h[:, 4:7] contains reactant atom coordinates.
            - h[:, 7:10] contains product atom coordinates.
        - The edge_index tensor contains pairs of atom indices representing edges in the graph.
        - The function calculates bond information based on atom types and distances, using
          predefined bond lengths for various atom combinations.
        - Bond types are determined by comparing distances between atoms with predefined bond lengths.
        - The function is currently hardcoded for hydrogen atoms and assumes no context.
    """
    atom_1, atom_2 = edge_index
    atom_encodings = h[:, :4]

    reactant = h[:, 4:7]
    product = h[:, 7:10]

    # Get the atom_1_atom_type and atom_2_atom_type:
    atom_1_type, atom_2_type = atom_encodings[atom_1], atom_encodings[atom_2]

    # Get Reactant Distance
    reactant_distance = reactant[atom_1] - reactant[atom_2]
    radial_reactant = torch.sum((reactant_distance) ** 2, 1).unsqueeze(1)
    reactant_distance = torch.sqrt(radial_reactant + 1e-12)

    # Get Product distance
    product_distance = product[atom_1] - product[atom_2]
    radial_product = torch.sum((product_distance) ** 2, 1).unsqueeze(1)
    product_distance = torch.sqrt(radial_product + 1e-12)

    # Get reactant_bonds:
    reactant_bonds = get_bond_type(
        atom_1_OHE=atom_1_type,
        atom_2_OHE=atom_2_type,
        distances=reactant_distance,  # noqa
    )
    product_bonds = get_bond_type(
        atom_1_OHE=atom_1_type,
        atom_2_OHE=atom_2_type,
        distances=product_distance,  # noqa
    )
    return reactant_bonds, product_bonds


# Get Bond Type function
def get_bond_type(atom_1_OHE, atom_2_OHE, distances):
    """# noqa
    This function takes two one-hot encoded representations of atoms (atom_1_OHE and atom_2_OHE)
    and a list of distances between atom pairs. It calculates the bond types between these atoms
    based on their types and the provided distances.

    Args:
        atom_1_OHE (torch.Tensor): One-hot encoded representation of the first atom.
        atom_2_OHE (torch.Tensor): One-hot encoded representation of the second atom.
        distances (list): List of distances between atom pairs.

    Returns:
        torch.Tensor: A tensor containing the bond types between the atom pairs.
                         Each element in the tensor represents a bond type: 1 for a bond present,
                         and 0 for no bond present.
    """
    # Create tensor the same size of distnace:
    bond_list = []

    # Bond OHE Dictionary:
    ohe_dict = {
        "C": [1, 0, 0, 0],
        "N": [0, 1, 0, 0],
        "O": [0, 0, 1, 0],
        "H": [0, 0, 0, 1],
    }

    # Bond Type and Distances: Check if both atom ohe are in the string -->
    # Distances taken from: https://cccbdb.nist.gov/diatomicexpbondx.asp
    # Units are Argnstroms:
    bond_dict = {
        "CC": 1.54,
        "NN": 1.098,
        "HH": 0.741,
        "OO": 1.208,
        "CH": 1.09,
        "CN": 1.47,
        "CO": 1.43,
        "NO": 1.36,
        "NH": 1.036,
        "OH": 0.970,
        "HC": 1.09,
        "NC": 1.47,
        "OC": 1.43,
        "ON": 1.36,
        "HN": 1.036,
        "HO": 0.970,
    }
    for atom_1, atom_2, distance in zip(
        atom_1_OHE.to(int).tolist(), atom_2_OHE.to(int).tolist(), distances
    ):  # noqa
        # CHeck if the distance is very small then return 0 for no bond present
        # This is most likely an overlap
        if distance < 1e-5:
            bond_list.append(0)

        # Check if we are dealing with valid atoms or padding:
        elif atom_1 in ohe_dict.values() and atom_2 in ohe_dict.values():
            # Get atom types
            atom_1_key = next(
                key for key, value in ohe_dict.items() if value == list(atom_1)  # noqa
            )
            atom_2_key = next(
                key for key, value in ohe_dict.items() if value == list(atom_2)  # noqa
            )

            # Is there a bond:
            bond_key = "".join([atom_1_key, atom_2_key])

            # Get bond length:
            true_bond_length = bond_dict.get(bond_key)

            # Now check if the distance is larger or smaller:
            bond_list.append(distance < true_bond_length)

        # Or if we have padding OHE:
        elif atom_1 == [0, 0, 0, 0] or atom_2 == [0, 0, 0, 0]:
            bond_list.append(1)  # Will removed by edge mask anyways
        # OR If we have a wrong OHE:
        else:
            raise ValueError("Invalid atom encoding provided.")
    # Turn bond info to tensor:
    bond_list = torch.tensor(bond_list).unsqueeze(1)

    return bond_list


if __name__ == "__main__":
    # data = pd.read_csv(
    #     "data/Dataset_W93/data/w93_dataset/wb97xd3.csv", index_col=0
    # )  # noqa

    # reactant = data.iloc[1].rsmi
    # print(reactant)

    dataset = W93_TS_coords_and_reacion_graph()

    batch_size = 64
    train_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True
    )  # noqa
    sample, node_mask, edge_features = next(iter(train_loader))
    edge_features = edge_features.view(-1, 2)
    print(node_mask.shape)
    print(edge_features.shape)
