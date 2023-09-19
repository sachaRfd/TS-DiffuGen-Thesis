from data.Dataset_W93.dataset_class import W93_TS
import os

"""
This script contains the pytorch dataset class for the
samples created from RDKIT.
"""


class RDKIT_dataset(W93_TS):
    def __init__(
        self,
        directory="data/Dataset_RDKIT/data/Clean_Geometries/",
        remove_hydrogens=False,
        graph=False,
        plot_distribution=False,
        include_context=False,
    ):
        super().__init__(
            directory,
            remove_hydrogens,
            graph,
            plot_distribution,
            include_context,
        )

    def load_data(self):
        """
        Loads the dataset by extracting data from reaction files and performing preprocessing steps.
        """  # noqa
        self.count_data()
        for reaction_number in self.reaction_numbers:
            self.extract_data(reaction_number)
        self.atom_count()

        if self.plot_distribution:
            self.plot_molecule_size_distribution()

        self.one_hot_encode()

        assert (
            self.reactant_padded.shape
            == self.product_padded.shape
            == self.transition_states_padded.shape  # noqa
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
        self.reaction_numbers = []
        for folder in os.listdir(self.directory):
            if folder.startswith("Reaction_"):
                reaction_num = folder[9:]

                self.reaction_numbers.append(reaction_num)
                self.count += 1

    def extract_data(self, reaction_number):
        """
        Extracts reactant, product, and transition state information from specified reaction.
        """  # noqa
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


if __name__ == "__main__":
    print("Running Script")
    dataset = RDKIT_dataset()
    print(len(dataset))
