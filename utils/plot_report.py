# Basic
import os
import numpy as np

# Plot utils
from matplotlib import cm
from matplotlib import pyplot as plt


def plot_report(arrays, labels, config, plot_mean=False, save_path=None, file_name=None) -> None:
    """
    Plot the performance report.
    :param arrays: The performance indications.
    :param labels: The labels of each curve.
    :param config: Plot configurations.
    :param plot_mean: Whether to plot the mean of each array.
    :param save_path: The directory to save file.
    :param file_name: The file name.
    """
    if not all(len(array) == len(arrays[0]) for array in arrays):
        raise ValueError("All arrays must be the same length.")

    plt.figure(figsize=(10, 6))
    iterations = [i for i in range(len(arrays[0]))]

    cmap = cm.get_cmap(config.get('colormap', 'tab20b'), len(arrays) if plot_mean else 3 * len(arrays))

    for i, (arr, label) in enumerate(zip(arrays, labels)):
        this_color = cmap(i)
        if plot_mean:
            appl_mean = np.mean(arr)                            # Application Mean: Include 0 values
            perf_mean = np.mean(arr, where=(arr > 1e-5))        # Performance Mean: Exclude 0 values

            # Plot the two means
            plt.plot(iterations, [appl_mean for _ in range(len(arr))],
                     linestyle='--', color=this_color, label=f"{label} - Appl Mean={appl_mean:.3f}")
            plt.plot(iterations, [perf_mean for _ in range(len(arr))],
                     linestyle=':',  color=this_color, label=f"{label} - Perf Mean={perf_mean:.3f}")

        # Plot the array itself
        plt.plot(iterations, arr, color=this_color, label=f"{label}")

    plt.title(config['title'])
    plt.xlabel(config['x_name'])
    plt.ylabel(config['y_name'])
    plt.legend()
    plt.grid(True)
    if save_path is not None and file_name is not None:
        plt.savefig(os.path.join(save_path, file_name))
    plt.show()
