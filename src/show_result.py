import pickle
import matplotlib
import matplotlib.axes
import matplotlib.cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np
import tyro
from evaluate import EvaluationResult, MethodType, shape_names


def main(resultdir: str, /, figname: str = "result/result.pdf", plot: bool = True):
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

    n_classes = len(shape_names)
    n_stiffness = len(stiffness_list)
    dt = 1e-3 * result.sequence_length

    fig = plt.figure(figsize=(12 * cm, 9 * cm), constrained_layout=True)
    gs_master = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.5, 1])
    gs1 = GridSpecFromSubplotSpec(nrows=2, ncols=3, subplot_spec=gs_master[0, 0])
    gs2 = GridSpecFromSubplotSpec(nrows=1, ncols=3, subplot_spec=gs_master[1, 0])

    axes_k: list[matplotlib.axes.Axes] = []
    axes_s: list[matplotlib.axes.Axes] = []
    for i in range(n_classes):
        ax = fig.add_subplot(gs1[0, i])
        ax.set_xticklabels([])

        if i == 0:
            ax.set_title("Cylinder")
            ax.set_ylabel("Stiffness [N/m]")
        if i == 1:
            ax.set_title("Box")
            ax.set_yticklabels([])
        if i == 2:
            ax.set_title("Sphere")
            ax.set_yticklabels([])

        axes_k.append(ax)

    for i in range(n_classes):
        ax = fig.add_subplot(gs1[1, i])
        ax.set_xlabel("Time [s]", labelpad=0)
        ax.set_ylim(0, 1.05)

        if i == 0:
            ax.set_ylabel("Probability\nof shape")
        if i == 1:
            ax.set_yticklabels([])
        if i == 2:
            ax.set_yticklabels([])

        axes_s.append(ax)

    ax_k = fig.add_subplot(gs2[0, 0])
    ax_k.set_aspect("equal")
    ax_k.set_xlabel("Estimated stiffness [N/m]")
    ax_k.set_ylabel("Actual stiffness [N/m]")

    ax_k2 = fig.add_subplot(gs2[0, 1])
    ax_k2.set_xlabel("Times [s]")
    ax_k2.set_ylabel("Standard deviation\n[N/m]")

    ax_s = fig.add_subplot(gs2[0, 2])
    ax_s.set_xlabel("Times [s]")
    ax_s.set_ylabel("Entropy of\nshape estimation")

    stiffness_plotted = {}
    for k in stiffness_list:
        stiffness_plotted[k] = {}
        for s in range(n_classes):
            stiffness_plotted[k][s] = False

    for i in range(result.n_data):
        k = int(np.mean(result.stiffness_true[i]))
        s = int(np.mean(result.shape_true[i]))
        idx_shape = int(np.mean(result.shape_true[i]))
        idx_stiffness = stiffness_list.index(k)

        if stiffness_plotted[k][s]:
            continue
        stiffness_plotted[k][s] = True

        times = np.arange(result.stiffness_true.shape[1]) * dt
        color = matplotlib.colormaps.get_cmap("viridis")(idx_stiffness / n_stiffness)
        axes_k[idx_shape].plot(
            times, result.stiffness_true[i].flatten(), ls="--", lw=1, color=color
        )
        axes_k[idx_shape].plot(
            times, result.stiffness_mu[i].flatten(), lw=1, color=color
        )
        axes_k[idx_shape].fill_between(
            times,
            result.stiffness_mu[i].flatten() - result.stiffness_sigma[i].flatten(),
            result.stiffness_mu[i].flatten() + result.stiffness_sigma[i].flatten(),
            color=color,
            alpha=0.3,
        )
        axes_k[idx_shape].set_xlim(times[0], times[-1])

        axes_s[idx_shape].plot(
            times, result.shape_est[i, :, idx_shape].flatten(), lw=1, color=color
        )
        axes_s[idx_shape].set_xlim(times[0], times[-1])

        ax_k.scatter(
            [result.stiffness_mu[i, -1]],
            [result.stiffness_true[i, -1]],
            s=3,
            color=color,
        )

    ymin_list = [ax.get_ylim()[0] for ax in axes_k]
    ymax_list = [ax.get_ylim()[1] for ax in axes_k]
    for ax in axes_k:
        ax.set_ylim(min(ymin_list), max(ymax_list))

    val_max = max(ax_k.get_xlim()[1], ax_k.get_ylim()[1])
    ax_k.plot([0, val_max], [0, val_max], lw=0.5, color="black", zorder=0)

    std = np.mean(np.sqrt(result.stiffness_sigma), axis=0)
    times = np.arange(result.stiffness_true.shape[1]) * dt
    ax_k2.plot(times, std)
    ax_k2.set_xlim(times[0], times[-1])
    ax_k2.set_ylim(ymin=0.0, ymax=20)

    std = np.mean(np.sqrt(result.stiffness_sigma), axis=0)
    entropy = np.mean(
        np.sum(
            -result.shape_est * np.log(result.shape_est) / np.log(n_classes), axis=2
        ),
        axis=0,
    )
    times = np.arange(result.stiffness_true.shape[1]) * dt
    ax_s.plot(times, entropy)
    ax_s.set_yscale("log")
    ax_s.set_xlim(times[0], times[-1])
    ax_s.set_ylim(ymin=4e-7, ymax=2e0)
    ax_s.set_yticks([1e0, 1e-2, 1e-4, 1e-6])

    fig.savefig(figname)

    if plot:
        plt.show()


if __name__ == "__main__":
    tyro.cli(main)
