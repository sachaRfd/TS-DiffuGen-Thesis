import os
import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

"""

This dataset class will handle the different generated samples
for the classification task.

Will read the .csv file and use those values as labels.

"""


class classifier_dataset(Dataset):
    def __init__(
        self,
        directory="data/Dataset_generated_samples/Clean/",
        csv_directory="data/Dataset_generated_samples/reaction_dmae_labels.csv",  # noqa
        number_of_samples=40,
        only_output_ts=False,
    ):
        super().__init__()
        assert os.path.exists(directory)
        assert os.path.exists(csv_directory)
        self.directory = directory

        # Read the CSV file:
        self.csv_file = pd.read_csv(csv_directory, index_col=0)

        self.number_of_samples = number_of_samples

        self.only_output_ts = only_output_ts

        # Data-structures:
        self.ohe_dict = {
            "C": [1, 0, 0, 0],
            "N": [0, 1, 0, 0],
            "O": [0, 0, 1, 0],
            "H": [0, 0, 0, 1],
        }

        self.atom_dict = {}
        self.data = []
        self.reactant = []
        self.product = []
        self.samples = []
        self.labels = []

        self.node_mask = []
        self.count = 0

        self.load_data()
        print("Finished creating the dataset.")

    def load_data(self):
        """
        Loads the dataset by extracting data from reaction files and performing preprocessing steps.
        """  # noqa
        self.count_data()
        for reaction_number in range(self.count):
            self.extract_data(reaction_number)

        # Collapse all the list of lists:
        self.reactant = [item for sublist in self.reactant for item in sublist]
        self.product = [item for sublist in self.product for item in sublist]
        self.samples = [item for sublist in self.samples for item in sublist]
        self.labels = [item for sublist in self.labels for item in sublist]

        self.atom_count()

        assert (
            len(self.reactant)
            == len(self.product)
            == len(self.samples)
            == len(self.labels)
        )

        self.one_hot_encode()

        assert (
            self.reactant_padded.shape
            == self.product_padded.shape
            == self.samples_padded.shape  # noqa
        )  # noqa

        print("Setting up the final arrays.")
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
                data.append(line.split())
        return data

    def extract_data(self, reaction_number):
        """
        Extracts reactant, product, and transition state information from specified reaction.
        """  # noqa
        # Get the Full path:
        path = os.path.join(self.directory, f"Reaction_{reaction_number}")
        assert os.path.exists(path)  # Assert the path Exists

        # Check that in the path there are the three files:
        assert (
            len(os.listdir(path)) >= 3 + self.number_of_samples
        ), "The folder is missing files."

        # Now we can extract the Reactant, Product, TS info 40 times:
        reactant_matrix_list = []
        product_matrix_list = []

        for file in os.listdir(path):
            if file.startswith("Reactant"):
                # Extract reactant matrix
                reactant_matrix = self.read_xyz_file(os.path.join(path, file))

                # Copy 40 times reactant_matrix and put it in list:
                reactant_matrix_list.extend(
                    [reactant_matrix] * self.number_of_samples
                )  # noqa
                self.reactant.append(reactant_matrix_list)
            elif file.startswith("Product"):
                # Extract product matrix
                product_matrix = self.read_xyz_file(os.path.join(path, file))

                # Copy 40 times reactant_matrix and put it in list:
                product_matrix_list.extend(
                    [product_matrix] * self.number_of_samples
                )  # noqa
                self.product.append(product_matrix_list)

        # Get the samples out:
        samples = []
        for i in range(self.number_of_samples):
            current_path = path + f"/Sample_{i}.xyz"
            ts_sample = self.read_xyz_file(current_path)
            samples.append(ts_sample)
        self.samples.append(samples)

        # Get the RMSE Labels:
        reaction_number = int(path.split("_")[-1])
        row_of_rmse = self.csv_file.iloc[reaction_number][
            : self.number_of_samples
        ].to_list()
        self.labels.append(row_of_rmse)

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
        """
        Pads molecule so that all have the same size
            and can be fed through batch_loader
        """
        data = [
            mol + [[0.0] * len(mol[0])] * (max_length - len(mol))
            for mol in data  # noqa
        ]  # noqa
        return torch.tensor(data)

    def one_hot_encode(self):
        """
        Performs one-hot encoding of atom types and prepares other data-processing.
        """  # noqa

        print("\nThe Atom Encoding is the following:\n")
        for atom, count in self.ohe_dict.items():
            print(f"\t{atom}: {count}")
        print()

        # Replace the atom str with OHE vector
        self.replace_atom_types_with_ohe_vectors(self.reactant)
        self.replace_atom_types_with_ohe_vectors(self.product)
        self.replace_atom_types_with_ohe_vectors(self.samples)

        # Convert everything in the self.reactants, self.products, and self.transition_states to floats:  # noqa
        self.reactant = self.convert_to_float(self.reactant)
        self.product = self.convert_to_float(self.product)
        self.samples = self.convert_to_float(self.samples)

        # Calculate the center of gravity and remove it
        self.delete_centre_gravity()

        # Convert everything in the self.reactant, self.product, and self.transition_states back to lists  # noqa
        self.reactant = self.convert_to_list(self.reactant)
        self.product = self.convert_to_list(self.product)
        self.samples = self.convert_to_list(self.samples)

        # # Check the maximum length among all nested lists for padding
        max_length = max(len(mol) for mol in self.reactant)

        # Pad the nested lists to have the same length
        self.reactant_padded = self.pad_data(
            self.reactant,
            max_length=max_length,
        )
        self.product_padded = self.pad_data(
            self.product,
            max_length=max_length,
        )
        self.samples_padded = self.pad_data(
            self.samples,
            max_length=max_length,
        )

        # Create the node mask:
        self.node_mask = torch.tensor(
            [
                [1.0] * len(mol) + [0.0] * (max_length - len(mol))
                for mol in self.reactant
            ]
        )

    def replace_atom_types_with_ohe_vectors(self, molecule_list):
        """
        Replaces atom types in molecule data with their one-hot encoded vectors.
        """  # noqa
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
        for index in range(self.count * self.number_of_samples):
            reactant_coords = np.array(self.reactant[index])[
                :, len(self.atom_dict) :  # noqa
            ].astype(float)
            product_coords = np.array(self.product[index])[
                :, len(self.atom_dict) :  # noqa
            ].astype(float)

            ts_coords = np.array(self.samples[index])[
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
            self.samples[index] = np.array(self.samples[index])
            self.samples[index][:, len(self.atom_dict) :] = (  # noqa
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
        """
        Creates data arrays with context information based on the selected context or without context.
        """  # noqa
        print("Not including Context information")
        for index in tqdm(range(self.count * self.number_of_samples)):
            if self.only_output_ts:
                x = torch.cat(
                    [
                        self.reactant_padded[index, :, :],
                        self.samples_padded[index, :, :],
                    ],
                    dim=1,
                )
            else:
                x = torch.cat(
                    [
                        self.reactant_padded[index, :, :],
                        self.product_padded[index, :, -3:],
                        self.samples_padded[index, :, -3:],
                    ],
                    dim=1,
                )
            self.data.append(x)

    def get(self, idx):
        """
        Retrieves a data sample and its corresponding node mask.
        """
        return self.data[idx], self.node_mask[idx], self.labels[idx]

    def len(self):
        """
        Returns the length of the dataset.
        """
        return len(self.data)


if __name__ == "__main__":
    print("Running Script")
    dataset = classifier_dataset(number_of_samples=10)

    # In all papers they use 8:1:1 ratio
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.2, random_state=42
    )
    val_dataset, test_dataset = train_test_split(
        test_dataset, test_size=0.5, random_state=42
    )

    batch_size = 32
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    sample, node_mask, label = next(iter(train_loader))
    print(sample.shape)
    print(sample)

    print(node_mask.shape)
    print(node_mask)

    print(label)
