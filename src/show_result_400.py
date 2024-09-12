import pickle
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import tyro
from evaluate import EvaluationResult, MethodType, shape_names


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

    stiffness_list = list(
        set([int(np.mean(result.stiffness_true[i])) for i in range(result.n_data)])
    )
    stiffness_list.sort()

    n_stiffness = len(stiffness_list)
    dt = 1e-3 * result.sequence_length

    fig = plt.figure(figsize=(4.5 * cm, 3.2 * cm), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Stiffness [N/m]")
    ax.set_ylim(0, 200)

    i = 104  # Index of a representative data to be shown
    print(result.stiffness_true[i, 0, 0], result.shape_true[i, 0, 0])

    k = int(np.mean(result.stiffness_true[i]))
    idx_stiffness = stiffness_list.index(k)
    color = matplotlib.colormaps.get_cmap("viridis")(idx_stiffness / n_stiffness)

    times = np.arange(result.stiffness_true.shape[1]) * dt
    ax.plot(times, result.stiffness_true[i].flatten(), ls="--", lw=1, color=color)
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
