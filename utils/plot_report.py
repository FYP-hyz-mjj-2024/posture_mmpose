from matplotlib import pyplot as plt
import numpy as np


def plot_report(arrays, labels, config, plot_mean=False) -> None:
    """
    Plot the performance report.
    :param arrays: The performance indications.
    :param labels: The labels of each curve.
    :param config: Plot configurations.
    """
    if not all(len(array) == len(arrays[0]) for array in arrays):
        raise ValueError("All arrays must be the same length.")

    plt.figure(figsize=(10, 6))
    iterations = [i for i in range(len(arrays[0]))]

    for arr, label in zip(arrays, labels):
        if plot_mean:
            mean = np.mean(arr)
            plt.plot(iterations, [mean for _ in range(len(arr))], linestyle='--', label=f"{label} - Mean={mean:.2f}")
        plt.plot(iterations, arr, label=f"{label}")

    plt.title(config['title'])
    plt.xlabel(config['x_name'])
    plt.ylabel(config['y_name'])
    plt.legend()
    plt.grid(True)
    plt.show()