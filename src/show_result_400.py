import pickle
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import tyro
from evaluate import EvaluationResult, shape_names


def main(resultdir: str, /, figname: str = "result/result_400.pdf", plot: bool = True):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}"
    plt.rcParams["font.family"] = "Latin Modern Roman"
    plt.rcParams["font.size"] = 8
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["axes.titlesize"] = "medium"
    cm = 1 / 2.54

    with open(resultdir, "rb") as fp:
        result: EvaluationResult = pickle.load(fp)

    n_classes = len(shape_names)
    n_stiffness = int(result.n_data / n_classes)
    dt = 1e-3 * result.sequence_length

    fig = plt.figure(figsize=(4.5 * cm, 3.2 * cm), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Stiffness [N/m]")
    ax.set_ylim(-50, 850)

    for i in range(result.n_data):
        # Here, it is assumed that e.g.:
        # stiffness: [1, 100, 200, ..., 1, 100, 200, ..., 1, 100, 200, ...]
        # shape: [0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2, ...]
        idx_shape = int(np.floor(i / n_stiffness))
        idx_stiffness = i % n_stiffness

        k_true = float(np.mean(result.stiffness_true[i]))

        if 350 < k_true < 450 and idx_shape == 0:
            print(k_true)
            times = np.arange(result.stiffness_true.shape[1]) * dt
            color = matplotlib.colormaps.get_cmap("viridis")(
                idx_stiffness / n_stiffness
            )
            ax.plot(
                times, result.stiffness_true[i].flatten(), ls="--", lw=1, color=color
            )
            ax.plot(times, result.stiffness_mu[i].flatten(), lw=1, color=color)
            ax.fill_between(
                times,
                result.stiffness_mu[i].flatten() - result.stiffness_sigma[i].flatten(),
                result.stiffness_mu[i].flatten() + result.stiffness_sigma[i].flatten(),
                color=color,
                alpha=0.3,
            )
            ax.set_xlim(times[0], times[-1])

    fig.savefig(figname)

    if plot:
        plt.show()


if __name__ == "__main__":
    tyro.cli(main)
