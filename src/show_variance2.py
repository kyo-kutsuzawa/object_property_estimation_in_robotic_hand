import pickle
import matplotlib.pyplot as plt
import numpy as np
import tyro
from evaluate import EvaluationResult, shape_names


def main(
    figname: str = "result/result_comparison.pdf",
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
        "result/result_test.pickle",
        "result/result_test_big.pickle",
        "result/result_test_noise_2_2.pickle",
        "result/result_test_noise_8_8.pickle",
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

    fig = plt.figure(figsize=(7 * cm, 5 * cm), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    for i, result in enumerate(results):
        stiffness = np.zeros((n_stiffness,))
        sigma_list = [[] for _ in range(n_stiffness)]
        for j in range(result.n_data):
            pass
            # Here, it is assumed that e.g.:
            # stiffness: [1, 100, 200, ..., 1, 100, 200, ..., 1, 100, 200, ...]
            # shape: [0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2, ...]
            idx_stiffness = j % n_stiffness

            stiffness[idx_stiffness] = np.mean(result.stiffness_true[j])
            sigma_list[idx_stiffness].extend(
                result.stiffness_sigma[j, idx_start:, :].flatten().tolist()
            )

        sigma_array = np.stack(sigma_list, axis=0)
        sigma_mean = np.mean(sigma_array, axis=1)
        ax.plot(
            stiffness,
            sigma_mean,
            label=labels[i],
            marker=markers[i],
            lw=1,
            markersize=5,
        )
    ax.set_xlim(xmin=0, xmax=800)
    ax.set_ylim(ymin=0, ymax=160)
    ax.set_xlabel("True stiffness [N/m]")
    ax.set_ylabel("Standard deviation [N/m]")
    ax.legend(ncol=2, loc="upper left")

    fig.savefig(figname)

    if plot:
        plt.show()


if __name__ == "__main__":
    tyro.cli(main)
