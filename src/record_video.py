from dataclasses import dataclass, field
import enum
import os
import pickle
from PIL import Image, ImageDraw
from subprocess import Popen, PIPE
import matplotlib
import matplotlib.axes
import matplotlib.cm
import matplotlib.figure
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import tyro
from myenv import HandEnv
from collect_dataset import BEFORE_TOUCH, MAX_ITER_PER_EP, OPEN_CLOSE_DIV, START_STEP

shape_names: list = ["Cylinder", "Box", "Sphere"]


class MethodType(enum.Enum):
    unspecified = enum.auto()
    baseline = enum.auto()
    proposed = enum.auto()


@dataclass
class EvaluationResult:
    stiffness_true: np.ndarray = field(default_factory=np.ndarray)
    shape_true: np.ndarray = field(default_factory=np.ndarray)
    stiffness_mu: np.ndarray = field(default_factory=np.ndarray)
    stiffness_sigma: np.ndarray = field(default_factory=np.ndarray)
    shape_est: np.ndarray = field(default_factory=np.ndarray)
    n_data: int = 0
    sequence_length: int = 0


@dataclass
class TrajectoryDataset:
    joint_angle: np.ndarray = field(default_factory=np.ndarray)
    joint_velocity: np.ndarray = field(default_factory=np.ndarray)
    joint_command: np.ndarray = field(default_factory=np.ndarray)
    stiffness: np.ndarray = field(default_factory=np.ndarray)
    shape_class: np.ndarray = field(default_factory=np.ndarray)
    object_position: np.ndarray = field(default_factory=np.ndarray)
    object_orientation: np.ndarray = field(default_factory=np.ndarray)
    length: int = 0


class VideoRecorder:
    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.ffmpeg_process = None

    def write_frame(
        self,
        env: HandEnv,
        fig: matplotlib.figure.Figure,
        stiffness: float,
        shape_class: int,
    ) -> None:
        env_array = env.viewer._read_pixels_as_in_window(resolution=(500, 300))

        fig.canvas.draw()
        data = fig.canvas.buffer_rgba()
        height, width = fig.canvas.get_width_height()
        fig_array = np.frombuffer(data, dtype=np.uint8).reshape(width, height, 4)

        frame_height, frame_width = 360, 750
        frame_array = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        frame_array[: env_array.shape[0], : env_array.shape[1], :] = env_array[:, :, :]
        frame_array[
            : fig_array.shape[0],
            env_array.shape[1] : env_array.shape[1] + fig_array.shape[1],
            :,
        ] = fig_array[:, :, :3]
        im_frame = Image.fromarray(frame_array)

        imd = ImageDraw.Draw(im_frame)
        text = f"Stiffness: {stiffness} N/m\nShape: {shape_names[shape_class]}"
        imd.text(
            (20, frame_height - 5),
            text,
            fill=(200, 200, 200),
            font_size=25,
            anchor="ld",
        )

        im_frame.save(self.ffmpeg_process.stdin, "png")

    def reset(self, outdir: str) -> None:
        if self.ffmpeg_process is not None:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()

        self.ffmpeg_process = Popen(
            [
                "ffmpeg",
                "-y",  # Overwrite output files without asking
                "-f",
                "image2pipe",
                "-r",  # Set frame rate (Hz value, fraction or abbreviation)
                str(self.fps),
                "-i",
                "-",
                "-r",
                str(self.fps),
                "-vcodec",
                "h264",
                "-pix_fmt",
                "yuv420p",
                "-b:v",  # Video bitrate
                "480k",
                "-loglevel",
                "warning",
                outdir,
            ],
            stdin=PIPE,
        )

    def finish(self) -> None:
        self.ffmpeg_process.stdin.close()
        self.ffmpeg_process.wait()
        self.ffmpeg_process = None


def main(
    resultdir: str,
    datasetdir: str,
    /,
    outdir: str = "video/video.mp4",
    big: bool = False,
):
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

    with open(datasetdir, "rb") as fp:
        data_dict: dict = pickle.load(fp)

    dataset = TrajectoryDataset(
        joint_angle=np.stack(data_dict["data"], axis=0)[:, :, 0:11],
        joint_velocity=np.stack(data_dict["data"], axis=0)[:, :, 11:22],
        joint_command=np.stack(data_dict["data"], axis=0)[:, :, 22:33],
        stiffness=np.stack(data_dict["stiffness"], axis=0),
        shape_class=np.stack(data_dict["shape"], axis=0),
        object_position=np.stack(data_dict["pos"], axis=0)[:, :3],
        object_orientation=np.stack(data_dict["pos"], axis=0)[:, 3:],
        length=len(data_dict["data"]),
    )

    print(dataset.object_position.shape, dataset.object_orientation.shape)

    dataset_type = 2
    dt = 1e-3 * result.sequence_length

    if big:
        models = [HandEnv.cylinder_big, HandEnv.box_big, HandEnv.ball_big]
    else:
        models = [HandEnv.cylinder, HandEnv.box, HandEnv.ball]

    fig = plt.figure(figsize=(6 * cm, 9 * cm), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=1)

    ax_k = fig.add_subplot(gs[0, 0])
    ax_k.set_ylabel("Stiffness [N/m]")
    ax_k.set_xticklabels([])
    ax_k.set_ylim(-50, 250)

    ax_s = fig.add_subplot(gs[1, 0])
    ax_s.set_xlabel("Time [s]", labelpad=0)
    ax_s.set_ylabel("Probability\nof shape")
    ax_s.set_ylim(0, 1.05)

    (plot_k,) = ax_k.plot([0, 0], [0, 0], ls="--", lw=1, color="C0")
    (plot_mu,) = ax_k.plot([0, 0], [0, 0], lw=1, color="C0")
    plot_sigma = ax_k.fill_between([0, 0], [0, 0], [0, 0], alpha=0.3, color="C0")
    plot_ss: list[matplotlib.lines.Line2D] = []
    for s in range(len(shape_names)):
        (l,) = ax_s.plot([0, 0], [0, 0], lw=1, color=f"C{s + 1}")
        plot_ss.append(l)

    env = HandEnv(
        sim_start=1,
        sim_step=1,
        num_envs=1,
        timestep=0.001,
        dataset=dataset_type,
        is_vis=True,
    )

    os.makedirs(os.path.dirname(outdir), exist_ok=True)
    recorder = VideoRecorder()

    recorder.reset(os.path.join(os.path.dirname(outdir), f"tmp.mp4"))
    episode = None
    env.load_env2()
    env.reset2(fluctuation=True)
    for t in range(2):
        env.step_touch(dataset_type, episode)
        recorder.write_frame(env, fig, 0, 0)
    recorder.finish()

    recorder.reset(outdir)
    for i in range(0, result.n_data, 10):
        stiffness = float(np.mean(result.stiffness_true[i]))
        shape_class = int(np.mean(result.shape_true[i]))

        episode = None
        env.model = models[shape_class]
        env.load_env2()
        env.set_new_stiffness2(stiffness)
        env.reset2(
            fluctuation=True,
            pos=dataset.object_position[i],
            quat=dataset.object_orientation[i],
        )

        env.viewer.cam.distance = 0.35
        env.viewer.cam.azimuth = 0

        times = np.arange(result.stiffness_true.shape[1]) * dt
        ax_k.set_xlim(times[0], times[-1])
        ax_s.set_xlim(times[0], times[-1])
        plot_k.set_data(times, result.stiffness_true[i, :, 0])

        for t in tqdm.tqdm(range(START_STEP + MAX_ITER_PER_EP)):
            if t < START_STEP:
                env.step_touch(dataset_type, episode)
            else:
                if t - START_STEP < BEFORE_TOUCH:
                    env.move(t - START_STEP, BEFORE_TOUCH)
                elif t - START_STEP >= BEFORE_TOUCH:
                    env.move(t - START_STEP, OPEN_CLOSE_DIV)
                env.step(dataset_type, episode)

            if t % result.sequence_length == 0:
                idx = int(np.floor(t / result.sequence_length))

                plot_mu.set_data(times[:idx], result.stiffness_mu[i, :idx, 0])
                plot_sigma.remove()
                plot_sigma = ax_k.fill_between(
                    times[:idx],
                    result.stiffness_mu[i, :idx, 0]
                    - result.stiffness_sigma[i, :idx, 0],
                    result.stiffness_mu[i, :idx, 0]
                    + result.stiffness_sigma[i, :idx, 0],
                    alpha=0.3,
                    color="C0",
                )
                for j, plot_s in enumerate(plot_ss):
                    plot_s.set_data(times[:idx], result.shape_est[i, :idx, j])

            if t % int(1 / recorder.fps * 1e3) == 0:
                recorder.write_frame(env, fig, stiffness, shape_class)

        for _ in range(int(recorder.fps * 0.5)):
            recorder.write_frame(env, fig, stiffness, shape_class)

    recorder.finish()


if __name__ == "__main__":
    tyro.cli(main)
