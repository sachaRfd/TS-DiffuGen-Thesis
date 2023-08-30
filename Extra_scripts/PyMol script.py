# Sacha Raffaud sachaRfd and acse-sr1022

import os
import pymol

# Start PyMOL and suppress its GUI
pymol.finish_launching(["pymol", "-q"])

# Set the background color to white
pymol.cmd.bg_color("white")

# Get a list of all directories in the current directory
directory_list = [dir for dir in os.listdir(".") if os.path.isdir(dir)]

for directory in directory_list:
    # Change to the current directory
    os.chdir(directory)

    # Get a list of all .xyz files in the folder
    file_list = [file for file in os.listdir(".") if file.endswith(".xyz")]

    for file in file_list:
        # Load the XYZ file
        pymol.cmd.load(file, "my_structure")

        pymol.cmd.hide("everything", "my_structure")

        # Set the representation style for atoms
        pymol.cmd.show("sticks", "my_structure")

        # Center the view
        pymol.cmd.center("my_structure")

        # Load the true_samples file
        true_samples_file = os.path.join(os.getcwd(), "true_sample.xyz")
        pymol.cmd.load(true_samples_file, "true_samples")
        pymol.cmd.hide("everything", "true_samples")
        pymol.cmd.show("sticks", "true_samples")

        # # Center the view
        pymol.cmd.center("all")

        # Colour the true sample and the generated one:
        pymol.cmd.color("green", "my_structure and element C")
        pymol.cmd.color("grey", "true_samples and element C")

        # Align my_structure with true_samples
        pymol.cmd.align("true_samples", "my_structure")

        # Calculate RMSD
        rmsd = pymol.cmd.rms_cur("true_samples", "my_structure")

        # Save the overlap image with RMSE value
        overlap_image_name = "overlap_{}_{}.png".format(
            os.path.splitext(file)[0], rmsd
        )  # noqa
        pymol.cmd.png(overlap_image_name, width=800, height=600, dpi=300, ray=1)  # noqa

        # Clear the palet and everything else for next iteration
        pymol.cmd.reinitialize()

    # Go back to the parent directory
    os.chdir("..")

for directory in directory_list:
    # Change to the current directory
    os.chdir(directory)

    # Get a list of all .xyz files in the folder
    file_list = [file for file in os.listdir(".") if file.endswith(".xyz")]

    # Set the color of carbon atoms to green
    pymol.cmd.color("green", "element C")
    pymol.cmd.show("sticks", "all")

    for file in file_list:
        # Load the XYZ file
        pymol.cmd.load(file, "my_structure")

        pymol.cmd.hide("everything", "my_structure")

        # Set the representation style for atoms
        pymol.cmd.show("sticks", "all")

        pymol.cmd.color("green", "element C")

        pymol.cmd.show("sticks", "my_structure")  # Try again

        # Center the view
        pymol.cmd.center("all")
        pymol.cmd.show("sticks", "all")  # Try again

        # Adjust the viewport
        pymol.cmd.viewport(800, 600)

        # Save an image of the structure
        image_name = os.path.splitext(file)[0] + ".png"
        pymol.cmd.png(image_name, width=800, height=600, dpi=300, ray=1)

        # Remove the loaded structure for the next iteration
        pymol.cmd.reinitialize()

    # Go back to the parent directory
    os.chdir("..")

# Quit PyMOL
pymol.cmd.quit()
