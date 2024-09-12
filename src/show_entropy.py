import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tyro
from evaluate import EvaluationResult, MethodType, shape_names


def main(
    dirname: str = "result",
    figname: str = "result_comparison_entropy.pdf",
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

    markers = ["o", "s", "v", "D"]

    results: list[EvaluationResult] = []

    for filename in filelist:
        with open(filename, "rb") as fp:
            results.append(pickle.load(fp))

    n_classes = len(shape_names)
    n_stiffness = int(results[0].n_data / n_classes)
    idx_start = 15

    fig = plt.figure(figsize=(7 * cm, 8 * cm), constrained_layout=True)
    ax_k = fig.add_subplot(2, 1, 1)
    ax_s = fig.add_subplot(2, 1, 2)

    for i, result in enumerate(results):
        stiffness_list = list(
            set([int(np.mean(result.stiffness_true[i])) for i in range(result.n_data)])
        )
        stiffness_list.sort()
        n_stiffness = len(stiffness_list)

        stiffness = np.array(stiffness_list)
        sigma_list = [[] for _ in range(n_stiffness)]
        entropy_list = [[] for _ in range(n_stiffness)]

        for j in range(result.n_data):
            k = int(np.mean(result.stiffness_true[j]))
            idx_stiffness = stiffness_list.index(k)

            sigma = result.stiffness_sigma[j, idx_start:, :]
            sigma_list[idx_stiffness].append(np.mean(sigma))

            p = result.shape_est[j, idx_start:, :]
            entropy = np.sum(-p * np.log(p) / np.log(n_classes), axis=1)
            entropy_list[idx_stiffness].append(np.mean(entropy))

        sigma_array = np.stack(sigma_list, axis=0)
        sigma_mean = np.mean(sigma_array, axis=1)
        ax_k.plot(
            stiffness,
            sigma_mean,
            color=f"C{i}",
            label=labels[i],
            marker=markers[i],
            lw=1,
            markersize=5,
        )

        entropy_array = np.stack(entropy_list, axis=0)
        entropy_mean = np.mean(entropy_array, axis=1)
        ax_s.plot(
            stiffness,
            entropy_mean,
            color=f"C{i}",
            label=labels[i],
            marker=markers[i],
            lw=1,
            markersize=5,
        )

    ax_k.set_xlim(xmin=0, xmax=200)
    ax_k.set_xticklabels([])
    ax_k.set_ylabel("Standard deviation [N/m]")
    ax_k.legend(ncol=2, loc="upper right", bbox_to_anchor=(1, 1.4))

    ax_s.set_xlim(xmin=0, xmax=200)
    ax_s.set_yscale("log")
    ax_s.set_xlabel("True stiffness [N/m]")
    ax_s.set_ylabel("Entropy of shape estimation")

    fig.savefig(os.path.join(dirname, figname))

    if plot:
        plt.show()


if __name__ == "__main__":
    tyro.cli(main)
