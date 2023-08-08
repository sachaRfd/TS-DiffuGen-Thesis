import h5py
import numpy as np
import os

""" # noqa
Script to iterate over the RGD1 Dataset 

- Download the RGD1_CHNO.h5 file and place it in the same directory as this one. 

Adapted from the following repository: https://zenodo.org/record/7618731



- Then run following script to create the .XYZ file
"""


def save_xyz_file(path, atoms, coordinates):
    with open(path, "w") as f:
        f.write(f"{len(atoms)}\n\n")
        for atom, coord in zip(atoms, coordinates):
            f.write(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")


def main():
    """
    Main Function
    """

    hf = h5py.File("data/Dataset_RGD1/RGD1_CHNO.h5", "r")

    # Convert number to symbol
    num2element = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}

    count = 0
    for Rind, Rxn in hf.items():
        print("Paring Reaction {}".format(Rind))

        # Parse smiles
        # Rsmiles, Psmiles = str(np.array(Rxn.get("Rsmiles"))), str(
        #     np.array(Rxn.get("Psmiles"))
        # )

        # Parse elements
        elements = [num2element[Ei] for Ei in np.array(Rxn.get("elements"))]

        # Parse geometries
        TS_G = np.array(Rxn.get("TSG"))
        R_G = np.array(Rxn.get("RG"))
        P_G = np.array(Rxn.get("PG"))

        # save_path = f"data/Dataset_RGD1/data/Clean_Geometries/Reaction_{count}/"  # noqa
        # os.makedirs(save_path, exist_ok=True)

        # # Save TS, reactant, and product geometries as .xyz files
        # save_xyz_file(
        #     os.path.join(save_path, "TS_geometry.xyz"), elements, TS_G
        # )  # noqa
        # save_xyz_file(
        #     os.path.join(save_path, "Reactant_geometry.xyz"), elements, R_G
        # )  # noqa
        # save_xyz_file(
        #     os.path.join(save_path, "Product_geometry.xyz"), elements, P_G
        # )  # noqa

        print(f"Geometries saved for Reaction {count}")
        count += 1
        exit()


if __name__ == "__main__":
    main()
