from dataclasses import dataclass, field
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np
import tyro


@dataclass
class TrajectoryDataset:
    joint_angle: np.ndarray = field(default_factory=np.ndarray)
    joint_velocity: np.ndarray = field(default_factory=np.ndarray)
    joint_command: np.ndarray = field(default_factory=np.ndarray)
    stiffness: np.ndarray = field(default_factory=np.ndarray)
    shape_class: np.ndarray = field(default_factory=np.ndarray)
    length: int = 0


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

    dataset = TrajectoryDataset(
        joint_angle=np.stack(data_dict["data"], axis=0)[:, :, 0:11],
        joint_velocity=np.stack(data_dict["data"], axis=0)[:, :, 11:22],
        joint_command=np.stack(data_dict["data"], axis=0)[:, :, 22:33],
        stiffness=np.stack(data_dict["stiffness"], axis=0),
        shape_class=np.stack(data_dict["shape"], axis=0),
        length=len(data_dict["data"]),
    )

    dt = 1e-3
    length = dataset.joint_angle.shape[1]
    n_dof = dataset.joint_angle.shape[2]
    shape_names = ["Cylinder", "Box", "Sphere"]

    fig = plt.figure(figsize=(12 * cm, 18 * cm), constrained_layout=True)
    gs = fig.add_gridspec(2, 1)
    n_row = int(np.ceil(np.sqrt(n_dof)))
    n_col = int(np.ceil(n_dof / n_row))
    gs_pos = GridSpecFromSubplotSpec(n_row, n_col, gs[0, 0])
    gs_vel = GridSpecFromSubplotSpec(n_row, n_col, gs[1, 0])
    axes_pos = [fig.add_subplot(gs_pos[*divmod(i, n_col)]) for i in range(n_dof)]
    axes_vel = [fig.add_subplot(gs_vel[*divmod(i, n_col)]) for i in range(n_dof)]

    ax_pos_dummy = fig.add_subplot(gs_pos[n_row - 1, n_col - 1])
    ax_vel_dummy = fig.add_subplot(gs_vel[n_row - 1, n_col - 1])

    labels = [
        "Index MP",
        "Index DP",
        "Index PIP",
        "Middle MP",
        "Middle DP",
        "Middle PIP",
        "Third MP",
        "Third DP",
        "Third PIP",
        "Thumb MP",
        "Thumb PIP",
    ]
    data_show = [
        (0, "-"),
        (490, "-"),
        (500, "-"),
        (990, "-"),
        (1000, "-"),
        (1490, "-"),
    ]

    times = np.arange(length) * dt

    for idx, label in data_show:
        print(int(dataset.shape_class[idx]), dataset.stiffness[idx])

        color = "C{}".format(int(dataset.shape_class[idx].item()) - 1)
        ls = "-" if dataset.stiffness[idx] < 100 else "--"
        lw = 1
        label = "{}, {} N/m".format(
            shape_names[int(dataset.shape_class[idx].item()) - 1],
            int(dataset.stiffness[idx]),
        )

        for i in range(n_dof):
            axes_pos[i].plot(
                times, dataset.joint_angle[idx, :, i], lw=lw, color=color, ls=ls
            )
            axes_vel[i].plot(
                times, dataset.joint_velocity[idx, :, i], lw=lw, color=color, ls=ls
            )

            if idx == 0:
                axes_pos[i].plot(
                    times,
                    dataset.joint_command[idx, :, i],
                    lw=0.5,
                    ls="-",
                    color="black",
                )

        ax_pos_dummy.plot([], [], lw=1, color=color, ls=ls, label=label)
        ax_vel_dummy.plot([], [], lw=1, color=color, ls=ls, label=label)

    ax_pos_dummy.plot([], [], lw=1, color="black", ls="-", label="Command")

    pos_max = max(
        max(dataset.joint_angle.flatten()), max(dataset.joint_command.flatten())
    )
    pos_min = min(
        min(dataset.joint_angle.flatten()), min(dataset.joint_command.flatten())
    )
    vel_max = max(dataset.joint_velocity.flatten())
    vel_min = min(dataset.joint_velocity.flatten())

    idx_left_bottom = n_row * (n_col - 1) + 1
    axes_pos[idx_left_bottom].set_xlabel("Time [s]", labelpad=-0.2)
    axes_vel[idx_left_bottom].set_xlabel("Time [s]", labelpad=-0.2)
    axes_pos[idx_left_bottom].set_ylabel(
        labels[idx_left_bottom] + " [rad]", labelpad=-0.2
    )
    axes_vel[idx_left_bottom].set_ylabel(
        labels[idx_left_bottom] + " [rad/s]", labelpad=-0.2
    )

    ax_pos_dummy.axis("off")
    ax_vel_dummy.axis("off")
    ax_pos_dummy.legend(fontsize=7)
    ax_vel_dummy.legend(fontsize=7)

    for i in range(n_dof):
        if i != idx_left_bottom:
            axes_pos[i].set_xticklabels([])
            axes_vel[i].set_xticklabels([])

            axes_pos[i].set_yticklabels([])
            axes_vel[i].set_yticklabels([])

            axes_pos[i].set_ylabel(labels[i], labelpad=-0.2)
            axes_vel[i].set_ylabel(labels[i], labelpad=-0.2)

        axes_pos[i].set_xlim(times[0], times[-1])
        axes_vel[i].set_xlim(times[0], times[-1])

        axes_pos[i].set_ylim(pos_min - 0.1, pos_max + 0.1)
        axes_vel[i].set_ylim(vel_min - 0.5, vel_max + 0.5)

    os.makedirs("result", exist_ok=True)
    fig.savefig("result/data_example.pdf")

    if plot:
        plt.show()


if __name__ == "__main__":
    tyro.cli(main)
