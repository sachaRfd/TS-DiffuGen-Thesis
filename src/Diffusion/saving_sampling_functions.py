# Sacha Raffaud sachaRfd and acse-sr1022

import torch


"""
This file contains the functions to convert diffusion samples from tensor 
format into writable .xyz format.

These functions are included within the PyTest framework.
"""  # noqa


def write_xyz_file(data, filename):
    """
    Writes Molecule Conformation data into an XYZ file in PyMol Format.# noqa

    Parameters:
        data (list of lists): A list containing the conformation data of the molecule.
            Each element in the list represents an atom in the molecule and is itself a list
            with four elements: atom name (str), x-coordinate (str), y-coordinate (str),
            and z-coordinate (str).
        filename (str): The name of the XYZ file to be created.

    Returns:
        None: This function does not return anything. It writes the data into the specified file.

    """

    # If the filename does not end with .xyz then add to it:
    if filename[-4:] != ".xyz":
        print(
            "Please make sure to have the .xyz extension to the file to be saved.\nFor now we will add it automatically."  # noqa
        )
        filename += ".xyz"

    with open(filename, "w") as f:
        # The first line should be the size of the molecule and the second line should be empty:# noqa
        f.write(str(len(data)) + "\n\n")
        for atom_list in data:
            line = str(
                atom_list[0]
                + " "
                + atom_list[1]
                + " "
                + atom_list[2]
                + " "
                + atom_list[3]
            )
            # print(line)
            f.write(line + "\n")


def return_xyz(sample, ohe_dictionary, remove_hydrogen=False):
    """
    Sets up sample into XYZ format.

    Parameters:
        sample (list of lists): A list of molecular structures (samples), where each sample is a list of atoms.# noqa
            Each atom is represented as a list with either 4 or 5 elements depending on the value of `remove_hydrogen`.
            If `remove_hydrogen` is False, each atom list contains the atom name (str), x-coordinate (float),
            y-coordinate (float), z-coordinate (float), and optionally an atom property (int).
            If `remove_hydrogen` is True, each atom list contains the atom name (str) and the three-dimensional
            coordinates (x, y, z) as floats.
        Dictionary with atom-encoding in there
        remove_hydrogen (bool, optional): A flag indicating whether hydrogen atoms should be removed from the output.
            If True, hydrogen atoms are excluded from the output, and the resulting XYZ format contains only
            non-hydrogen atoms. Default is False.

    Returns:
        list of lists: A list of molecular structures (samples) converted to the XYZ format.
            Each sample is a list of atoms, where each atom is represented as a list containing:
            - The atom name (str) obtained from the one-hot encoding dictionary.
            - The x-coordinate (float) of the atom's position in 3D space.
            - The y-coordinate (float) of the atom's position in 3D space.
            - The z-coordinate (float) of the atom's position in 3D space.
            If `remove_hydrogen` is False, the atom list also contains:
            - An atom property (int) obtained from the original atom data.
    """
    # Now we can remove the samples that have 0 in all the first 4 arrays
    clean_molecule = []
    for atom in sample[0]:
        atom_list = []
        if remove_hydrogen:
            atom_encoding = atom[:3].to(torch.int8)
        else:
            atom_encoding = atom[:4].to(torch.int8)
        if (
            list(atom_encoding) in ohe_dictionary.values()
        ):  # Check if we are dealing with real atom or just padding
            key = next(
                key
                for key, value in ohe_dictionary.items()
                if value == list(atom_encoding)
            )
            atom_list.append(key)
            if remove_hydrogen:
                coords = atom[3:].detach()
            else:
                coords = atom[4:].detach()
            atom_list.extend([str(coord.item()) for coord in coords])
            clean_molecule.append(atom_list)
    return clean_molecule
