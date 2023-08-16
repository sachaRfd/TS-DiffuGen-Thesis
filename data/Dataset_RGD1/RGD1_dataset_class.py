import os
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import erdos_renyi_graph
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


"""
Difference compared to other dataset class is that
when loading the files we exclude the first 2 lines
because they are the amount of atoms and an empty line.
"""


class RGD1_TS(Dataset):
    NUCLEAR_CHARGES = {"H": 1, "C": 6, "N": 7, "O": 8, "None": 0}
    VAN_DER_WAALS_RADIUS = {
        "H": 1.20,
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "None": 0,
    }

    def __init__(
        self,
        directory="data/Dataset_RGD1/data/Clean_Geometries/",
        remove_hydrogens=False,
        graph=False,
        plot_distribution=False,
        include_context=False,
    ):
        super().__init__()

        self.plot_distribution = plot_distribution
        self.graph = graph
        self.remove_hydrogens = remove_hydrogens
        self.directory = directory
        self.context = include_context
        self.count = 0

        # assert if there is context it should be either Nuclear, Van_Der_Waals or Activation_Energy  # noqa
        self.valid_contexts = [
            "Nuclear_Charges",
            "Van_Der_Waals",
            "Activation_Energy",
        ]
        assert (
            not self.context or self.context in self.valid_contexts
        ), "The context should be one of: " + ", ".join(self.valid_contexts)

        if self.context == "Activation_Energy":
            # Load the CSV file with extra variables:
            path_to_csv = "data/Dataset_W93/data/w93_dataset/wb97xd3.csv"
            # assert that the csv file is present in location:
            assert os.path.exists(
                path_to_csv
            ), f"CSV file '{path_to_csv}' does not exist"
            self.df = pd.read_csv(path_to_csv).ea

        # Data-structures:
        self.atom_dict = {}
        self.ohe_dict = {}
        self.data = []
        self.reactant = []
        self.product = []
        self.transition_states = []
        self.node_mask = []

        assert os.path.exists(self.directory)

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

        if self.plot_distribution:
            self.plot_molecule_size_distribution()

        self.one_hot_encode()

        assert (
            self.reactant.shape
            == self.product.shape
            == self.transition_states.shape  # noqa
        )  # noqa

        if self.graph:
            print("Creating Graphs")
            self.create_graph()
        else:
            print("Not Using graphs.")
            self.create_data_array()

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
                if self.remove_hydrogens and line[0] == "H":
                    continue
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
        assert len(os.listdir(path)) == 3, "The folder is missing files."

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

    def atom_count(self):
        """
        Counts the occurrences of different atom types in the reactions.
        """
        # Iterate over all the values in the lists to find if the different molecules:  # noqa
        for mol in self.reactant:
            for atom in mol:
                if atom[0] not in self.atom_dict:
                    self.atom_dict[atom[0]] = 1
                else:
                    self.atom_dict[atom[0]] += 1

    def plot_molecule_size_distribution(self):
        """
        Plots the distribution of molecule sizes in the dataset.
        """
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
        )  # Increase font size for x-axis label  # noqa
        plt.ylabel("Count", fontsize=15)  # Increase font size for y-axis label
        plt.title(
            "Distribution of Molecule Sizes", fontsize=16
        )  # Increase font size for title
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

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

    def create_graph(self):
        """
        Creates fully connected graphs from the data.
        """
        # We can append it to the self.data tensor of graphs, and make the positions the TS, we also make it fully connected  # noqa
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
        Creates data arrays with context information based on the selected context or without context.
        """
        if self.context:
            print(f"Including {self.context} to context")
            for index in range(self.count):
                ohes = self.reactant[index, :, :-3]

                if self.context == "Van_Der_Waals":
                    # Get the atom type
                    atom_types = [
                        self.get_keys_from_value(atom.tolist()) for atom in ohes  # noqa
                    ]  # Get the atom type
                    context_values = torch.tensor(
                        [
                            [self.VAN_DER_WAALS_RADIUS[atom_type[0]]]
                            for atom_type in atom_types
                        ]
                    )

                elif self.context == "Nuclear_Charges":
                    # Get the atom type
                    atom_types = [
                        self.get_keys_from_value(atom.tolist()) for atom in ohes  # noqa
                    ]
                    context_values = torch.tensor(
                        [
                            [self.NUCLEAR_CHARGES[atom_type[0]]]
                            for atom_type in atom_types
                        ]
                    )
                elif self.context == "Activation_Energy":
                    context_values = torch.tensor(
                        self.df.iloc[index], dtype=torch.float32
                    ).expand(  # noqa
                        self.reactant[0].shape[0], 1
                    )

                x = torch.cat(
                    [
                        ohes,
                        context_values,
                        self.reactant[index, :, -3:],
                        self.product[index, :, -3:],
                        self.transition_states[index, :, -3:],
                    ],
                    dim=1,
                )

                self.data.append(x)
        else:
            print("Not including Context information")
            for index in range(self.count):
                x = torch.cat(
                    [
                        self.reactant[index, :, :],
                        self.product[index, :, -3:],
                        self.transition_states[index, :, -3:],
                    ],
                    dim=1,
                )
                self.data.append(x)

    def get(self, idx):
        """
        Retrieves a data sample and its corresponding node mask.
        """
        return self.data[idx], self.node_mask[idx]

    def len(self):
        """
        Returns the length of the dataset.
        """
        return len(self.data)


if __name__ == "__main__":
    remove_hydrogens = False
    include_context = "Activation_Energy"
    dataset = RGD1_TS(
        directory="data/Dataset_RGD1/data/Clean_Geometries/",
        remove_hydrogens=remove_hydrogens,
        graph=False,
        plot_distribution=False,
        include_context=include_context,
    )

    # In all papers they use 8:1:1 ratio
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.2, random_state=42
    )
    val_dataset, test_dataset = train_test_split(
        test_dataset, test_size=0.5, random_state=42
    )

    batch_size = 20
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    print(next(iter(train_loader))[0][0][0])
