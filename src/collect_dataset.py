from argparse import ArgumentParser
import csv
import os
import pickle
from PIL import Image
import numpy as np
from myenv import HandEnv

NUM_EPISODES = 1
MAX_ITER_PER_EP = 2000
OPEN_CLOSE_DIV = 1600
START_STEP = 200
BEFORE_TOUCH = 200


def log_into_file(args):
    current_env = 0

    if args.snapshot:
        args.vis = True
        shapename = ["cylinder", "box", "sphere"]
        os.makedirs("snapshots", exist_ok=True)

    env_spec = HandEnv.get_std_spec(args)
    env = HandEnv(**env_spec)

    os.makedirs(args.data, exist_ok=True)
    path = os.path.join(args.data, "{}.pickle".format(args.data_name))
    file_csv = os.path.join(args.data, "{}.csv".format(args.data_name))
    data, stiffness, shape = list(), list(), list()

    episode = 1
    for ep in range(NUM_EPISODES * args.num_envs):
        if args.snapshot:
            if not current_env % 200 == 0:
                if ep % NUM_EPISODES == 0 and args.num_envs > 1:
                    current_env += 1
                episode += 1
                continue

        # change env number
        if ep % NUM_EPISODES == 0 and args.num_envs > 1:
            current_env += 1
            if current_env > args.num_envs:
                current_env = 0
            if (
                args.dataset == 0
                or args.dataset == 1
                or args.dataset == 2
                or args.dataset == 3
            ):
                env.load_env(current_env - 1)
            elif args.dataset == 5:
                env.load_env_random(current_env - 1)

        if (
            args.dataset == 0
            or args.dataset == 1
            or args.dataset == 2
            or args.dataset == 3
        ):
            current_object = env.reset(args.dataset, episode)
        elif args.dataset == 5:
            current_object = env.reset_random()

        print("Stiffness: {0[0]:5.1f}, Shape: {0[1]:.0f}".format(current_object))

        if args.snapshot:
            imgname = "snapshots/{}_{:03d}_{{:04d}}.png".format(
                shapename[int(current_object[1]) - 1], int(current_object[0])
            )

        # start squeezing an object
        samples = list()
        time = 0

        for _ in range(START_STEP):
            if args.snapshot:
                if time % 500 == 0:
                    img_array = env.viewer._read_pixels_as_in_window(
                        resolution=(500, 300)
                    )
                    img = Image.fromarray(img_array)
                    img.save(imgname.format(time))

            readings, contact = env.step_touch(args.dataset, episode)
            if args.mask_contact and not contact:
                readings = np.zeros_like(readings)
            if readings is not None:
                samples.append(readings)
            time += 1

        for i in range(MAX_ITER_PER_EP):
            if args.snapshot:
                if time % 500 == 0:
                    img_array = env.viewer._read_pixels_as_in_window(
                        resolution=(500, 300)
                    )
                    img = Image.fromarray(img_array)
                    img.save(imgname.format(time))
            else:
                env.render()

            if i < BEFORE_TOUCH:
                env.move(i, BEFORE_TOUCH)
            elif i >= BEFORE_TOUCH:
                env.move(i, OPEN_CLOSE_DIV)

            readings = env.step(args.dataset, episode)
            if readings is not None:
                samples.append(readings)
            time += 1

        # add to a pickle
        samples = np.array(samples)

        # Add noises
        if args.noise_pos > 0:
            samples[:, 0:11] += np.random.normal(0, np.sqrt(args.noise_pos))
        if args.noise_vel > 0:
            samples[:, 11:22] += np.random.normal(0, np.sqrt(args.noise_vel))

        data.append(samples)
        stiffness.append(current_object[0])
        shape.append(current_object[1])

        episode += 1

    file = open(path, "wb")
    pickle.dump({"data": data, "stiffness": stiffness, "shape": shape}, file)
    file.close()
    print("Total number of samples: {0}".format(len(data)))

    f_2 = open(file_csv, mode="w", newline="")
    writer_2 = csv.writer(f_2)
    for l in shape:
        writer_2.writerow([l])
    f_2.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sim-step", type=int, default=1)
    parser.add_argument("--timestep", type=int, default=0.001)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--mask-contact", type=bool, default=False)
    parser.add_argument("--sim-start", type=int, default=1)
    parser.add_argument(
        "--dataset", type=int, default=0
    )  # train:0, train_big:1, test:2, test_big:3
    parser.add_argument("--data", type=str, default="dataset")
    parser.add_argument("--data-name", type=str, default="data")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--noise-pos", type=float, default=0)
    parser.add_argument("--noise-vel", type=float, default=0)
    args, _ = parser.parse_known_args()
    log_into_file(args)
