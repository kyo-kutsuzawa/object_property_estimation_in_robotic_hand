import os
import pickle
import matplotlib.axes
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tyro


def main(filename: str = "dataset/train.pickle", /, plot: bool = True):
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}"
    plt.rcParams["font.family"] = "Latin Modern Roman"
    plt.rcParams["font.size"] = 8
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["axes.titlesize"] = "medium"
    cm = 1 / 2.54

    with open(filename, "rb") as f:
        data_dict: dict = pickle.load(f)

    fig = plt.figure(figsize=(7 * cm, 12 * cm), constrained_layout=True)
    axes: list[matplotlib.axes.Axes] = [fig.add_subplot(3, 1, i + 1) for i in range(3)]

    ylabels: list[str] = [
        "Joint angle [rad]",
        "Joint velocity [rad/s]",
        "Command [rad]",
    ]
    labels = [
        "F-MP",
        "F-DP",
        "F-PIP",
        "M-MP",
        "M-DP",
        "M-PIP",
        "R-MP",
        "R-DP",
        "R-PIP",
        "Th-MP",
        "Th-PIP",
    ]

    dt = 1e-3
    n_dof = 11
    idx_data = 0
    data: np.ndarray = data_dict["data"][idx_data]
    stiffness: float = data_dict["stiffness"][idx_data]
    shape: int = data_dict["shape"][idx_data]
    times = np.arange(data.shape[0]) * dt
    print(stiffness, shape)
    for i in range(len(axes)):
        for j in range(i * n_dof, (i + 1) * n_dof):
            if j - i * n_dof < 10:
                color = f"C{(j - i * n_dof)}"
            else:
                color = "black"
            axes[i].plot(
                times, data[:, j], label=labels[j - i * n_dof], lw=1, color=color
            )

        if i == len(axes) - 1:
            axes[i].set_xlabel("Time [s]")
            axes[i].legend(ncol=3, loc="upper right", bbox_to_anchor=(1, -0.3))
        else:
            axes[i].set_xticklabels([])

        axes[i].set_xlim(times[0], times[-1])
        axes[i].set_ylabel(ylabels[i])

    os.makedirs("result", exist_ok=True)
    fig.savefig("result/data_example.pdf")

    if plot:
        plt.show()


if __name__ == "__main__":
    tyro.cli(main)
