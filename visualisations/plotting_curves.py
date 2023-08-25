"""

Plot the alphas, sigmas

"""

import numpy as np
import matplotlib.pyplot as plt


def plot_data(timesteps, data, label, save_filename):
    plt.plot(timesteps, data, label=label)
    plt.xlabel("Timesteps")
    plt.ylabel(label.capitalize())
    plt.title(f"{label.capitalize()} vs. Timesteps")
    plt.legend()
    plt.savefig(save_filename)
    plt.show()


if __name__ == "__main__":
    folders = ["cosine", "sigmoid_2", "sigmoid_5"]

    alphas_data = {}
    sigmas_data = {}

    for folder in folders:
        path_alpha = f"plots_and_images/{folder}/alpha_t.npy"
        path_sigma = f"plots_and_images/{folder}/sigma_t.npy"

        alphas = np.load(path_alpha).squeeze()
        sigmas = np.load(path_sigma).squeeze()

        timesteps = np.linspace(0, alphas.shape[0], alphas.shape[0])

        alphas_data[folder] = alphas
        sigmas_data[folder] = sigmas

    # Plot alphas together
    plt.figure()
    for folder, alphas in alphas_data.items():
        plt.plot(timesteps, alphas, label=f"{folder} Alphas")
    plt.xlabel("Timesteps")
    plt.title("Alphas vs. Timesteps")
    plt.legend()
    # plt.savefig("plots_and_images/alphas_t.png")
    plt.show()

    # Plot sigmas together
    plt.figure()
    for folder, sigmas in sigmas_data.items():
        plt.plot(timesteps, sigmas, label=f"{folder} Sigmas")
    plt.xlabel("Timesteps")
    plt.title("Sigmas vs. Timesteps")
    plt.legend()
    # plt.savefig("plots_and_images/sigmas_t.png")
    plt.show()
