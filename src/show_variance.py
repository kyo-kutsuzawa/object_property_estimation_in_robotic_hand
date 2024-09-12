import os
import pickle
import matplotlib
import matplotlib.axes
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import tyro
from evaluate import EvaluationResult, MethodType


def main(
    dirname: str = "result",
    figname: str = "result_variance.pdf",
    plot: bool = True,
):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}"
    plt.rcParams["font.family"] = "Latin Modern Roman"
    plt.rcParams["font.size"] = 8
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["axes.titlesize"] = "medium"
    cm = 1 / 2.54

    filelist = [
        os.path.join(dirname, "result_test.pickle"),
        os.path.join(dirname, "result_test_big.pickle"),
        os.path.join(dirname, "result_test_noise_2_2.pickle"),
        os.path.join(dirname, "result_test_noise_8_8.pickle"),
    ]

    labels = [
        "i.i.d.",
        "bigger",
        "small noise",
        "large noise",
    ]

    results: list[EvaluationResult] = []

    for filename in filelist:
        with open(filename, "rb") as fp:
            results.append(pickle.load(fp))

    idx_start = 15

    fig = plt.figure(figsize=(14 * cm, 4 * cm), constrained_layout=True)
    axes: list[matplotlib.axes.Axes] = []
    for i in range(len(results)):
        ax = fig.add_subplot(1, len(results), i + 1)
        ax.set_aspect("equal")
        ax.set_ylabel("Standard deviation [N/m]")
        ax.set_xlabel("Error [N/m]")
        ax.set_title(labels[i])

        axes.append(ax)

    for i, result in enumerate(results):
        stiffness_list = list(
            set([int(np.mean(result.stiffness_true[i])) for i in range(result.n_data)])
        )
        stiffness_list.sort()
        n_stiffness = len(stiffness_list)

        error = abs(result.stiffness_mu - result.stiffness_true)[:, idx_start:, :]
        sigma = result.stiffness_sigma[:, idx_start:, :]
        corr = np.corrcoef(error.flatten(), sigma.flatten())[0, 1]
        print(f"Corr ({labels[i]}, total): {corr}")

        for j in range(result.n_data):
            k = int(np.mean(result.stiffness_true[j]))
            idx_stiffness = stiffness_list.index(k)

            color = matplotlib.colormaps.get_cmap("viridis")(
                idx_stiffness / n_stiffness
            )
            axes[i].scatter(error[j].flatten(), sigma[j].flatten(), color=color, s=5)

        range_max = max([np.max(error), np.max(sigma)]) * 1.05
        axes[i].set_xlim(0, range_max)
        axes[i].set_ylim(0, range_max)
        axes[i].plot([0, range_max], [0, range_max], lw=0.5, color="black", zorder=0)

    fig.savefig(os.path.join(dirname, figname))

    if plot:
        plt.show()


if __name__ == "__main__":
    tyro.cli(main)
