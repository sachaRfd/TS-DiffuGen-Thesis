# import numpy as np
import os
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import erdos_renyi_graph
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

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


class QM90_TS_reactant_coords_and_product_graph(Dataset):
    def __init__(
        self,
        directory="data/Dataset_W93/data/Clean_Geometries/",
        remove_hydrogens=False,
        graph=False,
        plot_distribution=False,
        include_context=False,
        graph_product=False,
    ):
        super().__init__()

        # First we can read all the data from the files:
        self.remove_hydrogen = remove_hydrogens
        self.directory = directory
        self.context = include_context
        self.count = 0

        # Assert if we are using graph_product not to have remove_hydrogens= True or inclde_cntext = True as implementation for them not yet finished   # noqa
        assert not (
            graph_product and (remove_hydrogens or include_context)
        ), "When using graph_product, remove_hydrogens and include_context must be set to False."  # noqa

        # Dictionaries
        self.atom_dict = {}
        self.ohe_dict = {}
        self.nuclear_charges = {
            "H": 1,
            "C": 6,
            "N": 7,
            "O": 8,
            "None": 0,
        }  # Dictionary containing the nuclear charges of the atoms
        self.van_der_waals_radius = {
            "H": 1.20,
            "C": 1.70,
            "N": 1.55,
            "O": 1.52,
            "None": 0,
        }  # Dictionary containing the van_der_waals_radius of the atoms in Argstroms   # noqa

        self.data = []
        self.reactant = []
        self.product = []
        self.transition_states = []

        # List for edge_attributes if we need them:
        self.product_graph = graph_product
        if self.product_graph:
            self.edge_attributes = []

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
        print(
            f"\nThe dataset includes {self.count} reactions and the following atom count:\n"  # noqa
        )
        for atom, count in self.atom_dict.items():
            print(f"\t{atom}: {count}")
        print()

        if plot_distribution:
            # Plot the size distribution of the molecules - Before we One hot encode:# noqa
            self.plot_molecule_size_distribution()

        # One Hot Encode the atoms:
        self.one_hot_encode()

        # Assert that the shapes are correct:
        assert (
            self.reactant.shape
            == self.product.shape
            == self.transition_states.shape  # noqa
        )  # noqa

        if graph:
            # Create the graphs:
            print("Creating Graphs")
            self.create_graph()
        else:
            print("Not Using graphs")
            self.create_data_array()

        if graph_product:
            # integrate the bond information as edge features and return them
            self.setup_edge_attributes()

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
        assert (
            len(os.listdir(path)) == 4
        ), "The folder is missing files."  # 4 FIles as we have the reactant images also in the directory   # noqa

        # Now we can extract the Reactant, Product, TS info:
        for file in os.listdir(path):  #
            if file.startswith("Reactant"):
                # Append to Reactant Matrix:
                reactant_matrix = []
                with open(os.path.join(path, file), "r") as read_file:
                    lines = read_file.readlines()
                    for line in lines[2:]:
                        if self.remove_hydrogen:
                            # Check that the line is not Oxygens:
                            if line[0] != "H":
                                reactant_matrix.append(line.split())
                        else:
                            reactant_matrix.append(line.split())
                self.reactant.append(reactant_matrix)

            elif file.startswith("Product"):
                # Append to Reactant Matrix:
                product_matrix = []
                with open(os.path.join(path, file), "r") as read_file:
                    lines = read_file.readlines()
                    for line in lines[2:]:
                        if self.remove_hydrogen:
                            if line[0] != "H":
                                product_matrix.append(line.split())
                        else:
                            product_matrix.append(line.split())
                self.product.append(product_matrix)

            elif file.startswith("TS"):
                # Append to Reactant Matrix:
                ts_matrix = []
                with open(os.path.join(path, file), "r") as read_file:
                    lines = read_file.readlines()
                    for line in lines[2:]:
                        if self.remove_hydrogen:
                            if line[0] != "H":
                                ts_matrix.append(line.split())
                        else:
                            ts_matrix.append(line.split())
                self.transition_states.append(ts_matrix)

    def atom_count(self):
        # Iterate over all the values in the lists to find if the different molecules:# noqa
        for mol in self.reactant:
            for atom in mol:
                if atom[0] not in self.atom_dict:
                    self.atom_dict[atom[0]] = 1
                else:
                    self.atom_dict[atom[0]] += 1

    def plot_molecule_size_distribution(self):
        # Count the size of each molecule
        molecule_sizes = [
            len(mol)
            for mol in self.reactant + self.product + self.transition_states  # noqa
        ]

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
        plt.xlabel(
            "Molecule Size", fontsize=15
        )  # Increase font size for x-axis label# noqa
        plt.ylabel("Count", fontsize=15)  # Increase font size for y-axis label
        plt.title(
            "Distribution of Molecule Sizes", fontsize=16
        )  # Increase font size for title
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def one_hot_encode(self):
        num_of_atoms = len(self.atom_dict)

        for index, atom in enumerate(self.atom_dict):
            ohe_vector = [0] * num_of_atoms
            ohe_vector[
                index
            ] = 1  # Set the position corresponding to the atom index to 1
            self.ohe_dict[atom] = ohe_vector

        print("\nThe Atom Encoding is the following:\n")
        for atom, count in self.ohe_dict.items():
            print(f"\t{atom}: {count}")
            # print(type(count))
        print()

        # Replace the atom str with OHE vector
        self.replace_atom_types_with_ohe_vectors(self.reactant)
        self.replace_atom_types_with_ohe_vectors(self.product)
        self.replace_atom_types_with_ohe_vectors(self.transition_states)

        # Convert everything in the self.reactants, self.products, and self.transition_states to floats:    # noqa
        self.reactant = [
            [[float(value) for value in atom] for atom in mol]
            for mol in self.reactant  # noqa
        ]
        self.product = [
            [[float(value) for value in atom] for atom in mol]
            for mol in self.product  # noqa
        ]
        self.transition_states = [
            [[float(value) for value in atom] for atom in mol]
            for mol in self.transition_states
        ]

        # Calculate the center of gravity and remove it
        self.delete_centre_gravity()  # This has been tested in the testing images on local computer --> WORKS# noqa

        # Convert everything in the self.reactant, self.product, and self.transition_states back to lists# noqa
        self.reactant = [mol.tolist() for mol in self.reactant]
        self.product = [mol.tolist() for mol in self.product]
        self.transition_states = [
            mol.tolist() for mol in self.transition_states
        ]  # noqa

        # Check the maximum length among all nested lists for padding
        max_length = max(
            len(mol)
            for mol in self.reactant + self.product + self.transition_states  # noqa
        )

        # Pad the nested lists to have the same length
        padded_reactant = [
            mol + [[0.0] * len(mol[0])] * (max_length - len(mol))
            for mol in self.reactant
        ]
        padded_product = [
            mol + [[0.0] * len(mol[0])] * (max_length - len(mol))
            for mol in self.product
        ]
        padded_transition_states = [
            mol + [[0.0] * len(mol[0])] * (max_length - len(mol))
            for mol in self.transition_states
        ]

        # Make a copy of the unpadded chemicals
        self.unpadded_chemical = self.reactant.copy()

        # Create the node mask:
        self.node_mask = torch.tensor(
            [
                [1.0] * len(mol) + [0.0] * (max_length - len(mol))
                for mol in self.reactant
            ]
        )

        # Assign the padded values to self.reactant, self.product, and self.transition_states# noqa
        self.reactant = torch.tensor(padded_reactant)
        self.product = torch.tensor(padded_product)
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

    def create_graph(self):
        # We can append it to the self.data tensor of graphs, and make the positions the TS, we also make it fully connected# noqa
        for index in range(self.count):
            # Create Fully connected graph
            num_nodes = self.reactant[index].shape[0]
            edge_index = erdos_renyi_graph(num_nodes, edge_prob=1.0)

            graph = Data(
                x=torch.cat(
                    [
                        self.reactant[index, :, :],
                        self.product[index, :, -3:],
                        self.transition_states[index, :, -3:],
                    ],
                    dim=1,
                ),
                pos=self.transition_states[index, :, -3:],
                edge_index=edge_index,
            )
            self.data.append(graph)

    def get_keys_from_value(self, search_value):
        result = [
            key for key, value in self.ohe_dict.items() if value == search_value  # noqa
        ]  # noqa
        return (
            result if result else ["None"]
        )  # Make it return none if it comes upon the masked atoms

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
                atom_types = [
                    self.get_keys_from_value(atom.tolist())
                    for atom in atomic_feature_vector
                ]  # Get the atom type
                van_der_waals = torch.tensor(
                    [
                        [self.van_der_waals_radius[atom_type[0]]]
                        for atom_type in atom_types
                    ]
                )  # Get the nuclear charge from the atom type

                # concatenate the OHE, the Nuclear charge, and then the coordinates of the reactant/product/TS# noqa
                x = torch.cat(
                    [
                        ohes,
                        van_der_waals,
                        self.reactant[index, :, -3:],
                        self.product[index, :, -3:],
                        self.transition_states[index, :, -3:],
                    ],
                    dim=1,
                )
                self.data.append(x)

        else:
            print("Not including Context information")

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
        edge_index = get_adj_matrix_no_batch(n_nodes=23)
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

        print("Done setting up edge attributes.")

    def get(self, idx):
        if self.product_graph:
            return (
                self.data[idx],
                self.node_mask[idx],
                self.edge_attributes[idx],
            )  # noqa

        else:
            return self.data[idx], self.node_mask[idx]

    def len(self):
        return len(self.data)


_edges_dict = {}


def get_adj_matrix_no_batch(n_nodes):
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
    """

    FOR NOW ONLY USING Reactant BOND

    Essentially creates the reaction graph from the reactant/product
     bond information

    x has shape [23, 3]
    --> We basically want to return

    row: 0,0,0,0 ect ect 22,22,22,22
    col: 0,1,2,3,4,5 ect ect 0,1,2,3, ect ect


    Function is hardcoded for usage with hydrogen and no Context
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
    """

    Return Bond type from atom type and disatnce between the two --> Currently only using single bonds# noqa

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
        if distance < 1e-5:
            bond_list.append(0)

        # Check if we are dealing with valide atoms or padding:
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

        else:
            bond_list.append(1)
    # Turn bond info to tensor:
    bond_list = torch.tensor(bond_list).unsqueeze(1)

    return bond_list


if __name__ == "__main__":
    # data = pd.read_csv(
    #     "data/Dataset_W93/data/w93_dataset/wb97xd3.csv", index_col=0
    # )  # noqa

    # reactant = data.iloc[1].rsmi
    # print(reactant)

    dataset = QM90_TS_reactant_coords_and_product_graph(graph_product=True)

    batch_size = 64
    train_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True
    )  # noqa
    sample, node_mask, edge_features = next(iter(train_loader))
    edge_features = edge_features.view(-1, 2)
    print(node_mask.shape)
    print(edge_features.shape)
