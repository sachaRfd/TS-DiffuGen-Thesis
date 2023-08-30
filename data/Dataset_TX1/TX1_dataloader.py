# Sacha Raffaud sachaRfd and acse-sr1022

import h5py
import numpy as np
import torch


"""
# noqa

This dataloader class was adapted from the following script: 
https://gitlab.com/matschreiner/Transition1x/-/blob/main/transition1x/dataloader.py?ref_type=heads


This script is used to read the TX1 Dataset .h5 file, and load the samples into a 
dataloader. This dataloader is then used downstream with a pytorch dataset class.


Not really testable without adding a .h5 file to Repo. (Which could overwhelm the project)

When this file is called the dataloader is instanciated and the length of the dataset is returned.

"""


REFERENCE_ENERGIES = {
    1: -13.62222753701504,
    6: -1029.4130839658328,
    7: -1484.8710358098756,
    8: -2041.8396277138045,
    9: -2712.8213146878606,
}

ohe_to_atomic_number = {
    1: [0, 0, 0, 1],
    6: [1, 0, 0, 0],
    7: [0, 1, 0, 0],
    8: [0, 0, 1, 0],
}


def get_molecular_reference_energy(atomic_numbers):
    """
    Calculate the molecular reference energy based on atomic numbers.

    This function takes a list of atomic numbers and calculates the molecular
    reference energy by summing up the individual reference energies of the atoms.

    Parameters:
        atomic_numbers (list): List of atomic numbers of the atoms in the molecule.

    Returns:
        float: The calculated molecular reference energy.
    """  # noqa
    molecular_reference_energy = 0
    for atomic_number in atomic_numbers:
        molecular_reference_energy += REFERENCE_ENERGIES[atomic_number]

    return molecular_reference_energy


def generator(formula, rxn, grp):
    """# noqa
    Iterate through an HDF5 group, processing energy and force data.

    This generator function iterates through an HDF5 group containing energy,
    force, atomic numbers, and position data for a molecular system. It processes
    this data to calculate various properties and returns dictionaries with the
    calculated information for each step.

    Parameters:
        formula (str): Chemical formula of the molecule.
        rxn (str): Reaction identifier or description.
        grp (h5py.Group): The HDF5 group containing the data.

    Returns:
        dict: A dictionary containing calculated properties for each step.
    """
    energies = grp["wB97x_6-31G(d).energy"]
    forces = grp["wB97x_6-31G(d).forces"]
    atomic_numbers = list(grp["atomic_numbers"])
    positions = grp["positions"]
    molecular_reference_energy = get_molecular_reference_energy(atomic_numbers)

    for energy, force, positions in zip(energies, forces, positions):
        d = {
            "rxn": rxn,
            "wB97x_6-31G(d).energy": energy.__float__(),
            "wB97x_6-31G(d).atomization_energy": energy
            - molecular_reference_energy.__float__(),
            "wB97x_6-31G(d).forces": force.tolist(),
            "positions": positions,
            "formula": formula,
            "atomic_numbers": atomic_numbers,
        }

        yield d


class Dataloader:
    """
    Iterates through an HDF5 dataset, providing access to reaction data.

    This class allows iteration through an HDF5 dataset containing reaction data.
    It provides the ability to loop through different configurations for each
    reaction and return the data in dictionaries. The class can also be configured
    to iterate only through reactants, products, and transition states.

    Parameters:
        hdf5_file (str): Path to the HDF5 file containing the dataset.
        datasplit (str, optional): Specifies the data split to iterate through (e.g., "data", "train", "val", "test"). (default: "data")
        only_final (bool, optional): If True, the iterator will only loop through reactants, products, and transition states. (default: False)

    Attributes:
        hdf5_file (str): Path to the HDF5 file containing the dataset.
        only_final (bool): Flag indicating whether to iterate only through reactants, products, and transition states.
        datasplit (str): Specifies the data split being iterated through.

    Returns:
        dict: A dictionary containing data for each configuration.
    """  # noqa

    def __init__(self, hdf5_file, datasplit="data", only_final=False):
        self.hdf5_file = hdf5_file
        self.only_final = only_final

        self.datasplit = datasplit
        if datasplit:
            assert datasplit in [
                "data",
                "train",
                "val",
                "test",
            ], "datasplit must be one of 'all', 'train', 'val' or 'test'"

    def __iter__(self):
        """
        Iterate through the HDF5 dataset and yield configuration data.
        """
        with h5py.File(self.hdf5_file, "r") as f:
            split = f[self.datasplit]

            for formula, grp in split.items():
                for rxn, subgrp in grp.items():
                    reactant = next(generator(formula, rxn, subgrp["reactant"]))  # noqa
                    product = next(generator(formula, rxn, subgrp["product"]))

                    if self.only_final:
                        transition_state = next(
                            generator(formula, rxn, subgrp["transition_state"])
                        )

                        yield {
                            "rxn": rxn,
                            "reactant": reactant,
                            "product": product,
                            "transition_state": transition_state,
                        }
                    else:
                        yield reactant
                        yield product
                        for molecule in generator(formula, rxn, subgrp):
                            yield molecule


if __name__ == "__main__":
    print("Running script")

    dataloader = Dataloader(
        "data/Dataset_TX1/Transition1x.h5", "test", only_final=True
    )  # noqa
    test_set = []
    test_node_masks = []
    max_length = 23  # For padding purposes (when hydrogens are present)
    for molecule in dataloader:
        # Atom type is in atomic number, lets convert it back to a OHE representation:# noqa
        atom_type = molecule["transition_state"]["atomic_numbers"]

        # Get the OHE Representations:
        ohe = []
        for atom in atom_type:
            ohe.append(ohe_to_atomic_number.get(atom))

        # Get the coordinates:
        ts_coordinates = molecule["transition_state"]["positions"]
        r_coordinates = molecule["reactant"]["positions"]
        p_coordinates = molecule["product"]["positions"]

        # Now we can setup a molecule list that will be fed through a downstream sampling model:# noqa
        mol = []
        for ohe_atom, reactant, product, ts in zip(
            ohe, r_coordinates, p_coordinates, ts_coordinates
        ):
            combined_values = [*ohe_atom, *reactant, *product, *ts]
            mol.append(combined_values)

        # Check that the mean of each variable is 0:
        mol = torch.tensor(mol)

        # Remove the mean from the reactants, products and transition states:
        mol[:, 4:7] -= mol[:, 4:7].mean(axis=0)
        mol[:, 7:10] -= mol[:, 7:10].mean(axis=0)
        mol[:, 10:13] -= mol[:, 10:13].mean(axis=0)

        # Check that the mean of each variable is 0:
        assert np.allclose(mol[:, 4:7].mean(axis=0), 0, atol=1e-5)
        assert np.allclose(mol[:, 7:10].mean(axis=0), 0, atol=1e-5)
        assert np.allclose(mol[:, 10:13].mean(axis=0), 0, atol=1e-5)

        # Create the padding:
        padding = torch.tensor(
            [[0.0] * mol[0].shape[0]] * (max_length - mol.shape[0])
        )  # noqa

        # Concatenate the padding to the original molecule tensor:
        padded_mol = torch.concatenate((mol, padding), dim=0)

        # Now create the node mask:
        node_mask = torch.tensor(
            [1.0] * mol[0].shape[0] + [0.0] * (max_length - mol.shape[0])
        )

        # Append them to dataset:
        test_node_masks.append(node_mask)
        test_set.append(padded_mol)

    print(f"Length of the test set is {len(test_set)}")
