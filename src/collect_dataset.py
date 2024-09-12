from argparse import ArgumentParser
import concurrent.futures
import os
import pickle
from PIL import Image
import time
import numpy as np
from myenv import HandEnv

MAX_ITER_PER_EP = 2000
OPEN_CLOSE_DIV = 1600
START_STEP = 200
BEFORE_TOUCH = 200


def main(args):
    env_spec = HandEnv.get_std_spec(args)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = []
        current_env = 0

        for ep in range(args.num_episodes * args.num_envs):
            if args.snapshot:
                if not (current_env % 200 == 0 or current_env % 200 == 199):
                    if ep % args.num_episodes == 0 and args.num_envs > 1:
                        current_env += 1
                    continue

            # change env number
            if ep % args.num_episodes == 0 and args.num_envs > 1:
                current_env += 1
                if current_env > args.num_envs:
                    current_env = 0

            result = executor.submit(run_rollout, args, env_spec, ep, current_env)
            futures.append(result)

        data = list()
        stiffness = list()
        shape = list()
        pos = list()

        for result in futures:
            values = result.result()
            data.append(values[0])
            stiffness.append(values[1])
            shape.append(values[2])
            pos.append(values[3])

        print("Total number of samples: {0}".format(len(data)))

        dataset = {"data": data, "stiffness": stiffness, "shape": shape, "pos": pos}
        filename = os.path.join(args.data, "{}.pickle".format(args.data_name))
        os.makedirs(args.data, exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(dataset, f)


def run_rollout(args, env_spec, ep, current_env):
    # Without resetting a random seed, the same values will be generated.
    # Each process has a unique pid, which ensures different random values between processes.
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    if args.snapshot:
        args.vis = True
        shapename = ["cylinder", "box", "sphere"]
        os.makedirs("snapshots", exist_ok=True)

    env = HandEnv(**env_spec)

    if args.snapshot:
        if not (current_env % 200 == 0 or current_env % 200 == 199):
            if ep % args.num_episodes == 0 and args.num_envs > 1:
                current_env += 1
            return

    # change env number
    if args.dataset in [0, 1, 2, 3, 4]:
        env.load_env(current_env - 1)
    elif args.dataset == 5:
        env.load_env_random(current_env - 1)

    if args.dataset in [0, 1, 2, 3, 4]:
        current_object = env.reset(args.dataset, current_env)
    elif args.dataset == 5:
        current_object = env.reset_random()

    if args.dataset in [20, 22, 23, 24]:
        # Training data (normal size, soft)
        if env.dataset == 20:
            n_stiffness = 50
            idx_shape, idx_stiffness = divmod((current_env - 1), n_stiffness)
            models_list = [env.cylinder, env.box, env.ball]
            env.model = models_list[idx_shape]
            stiffness = 1 + 4 * idx_stiffness

        # Training data (bigger size, soft)
        if env.dataset == 21:
            n_stiffness = 50
            idx_shape, idx_stiffness = divmod((current_env - 1), n_stiffness)
            models_list = [env.cylinder_big, env.box_big, env.ball_big]
            env.model = models_list[idx_shape]
            stiffness = 1 + 4 * idx_stiffness

        # Test data (normal size, soft)
        if env.dataset == 22:
            n_stiffness = 7
            idx_shape, idx_stiffness = divmod((current_env - 1), n_stiffness)
            models_list = [env.cylinder, env.box, env.ball]
            env.model = models_list[idx_shape]
            stiffness = 3 + 32 * idx_stiffness

        # Test data (bigger size, soft)
        if env.dataset == 23:
            n_stiffness = 7
            idx_shape, idx_stiffness = divmod((current_env - 1), n_stiffness)
            models_list = [env.cylinder_big, env.box_big, env.ball_big]
            env.model = models_list[idx_shape]
            stiffness = 3 + 32 * idx_stiffness

        # Training data (all, soft)
        if env.dataset == 24:
            n_stiffness = 50
            idx_shape, idx_stiffness = divmod((current_env - 1), n_stiffness)
            models_list = [
                env.cylinder,
                env.box,
                env.ball,
                env.cylinder_big,
                env.box_big,
                env.ball_big,
                env.cylinder_small,
                env.box_small,
                env.ball_small,
            ]
            env.model = models_list[idx_shape]
            stiffness = 1 + 4 * idx_stiffness

        current_object = env.load_env2()
        env.set_new_stiffness2(stiffness)
        obj_pos = env.reset2(fluctuation=True)

        if env.model in [
            env.cylinder,
            env.cylinder_big,
            env.cylinder_small,
            env.cylinder_pos,
            env.cylinder_middle,
            env.cylinder_pos_test,
        ]:
            current_shape = 1  # cylinder
        elif env.model in [
            env.box,
            env.box_big,
            env.box_small,
            env.box_pos,
            env.box_middle,
            env.box_pos_test,
        ]:
            current_shape = 2  # box
        elif env.model in [
            env.ball,
            env.ball_big,
            env.ball_small,
            env.ball_pos,
            env.ball_middle,
            env.ball_pos_test,
        ]:
            current_shape = 3  # ball
        labels = [stiffness, current_shape] + obj_pos.flatten().tolist()
        current_object = np.asanyarray(labels, dtype=float).reshape(-1)

    print(
        "ep: {:4d}, current_env: {:4d}, Stiffness: {:5.1f}, Shape: {:.0f}, Pos: {}, Quat: {}".format(
            ep,
            current_env,
            current_object[0],
            current_object[1],
            current_object[2:5],
            current_object[5:],
        )
    )

    if args.snapshot:
        imgname = "snapshots/{}_{:03d}_{{:04d}}.png".format(
            shapename[int(current_object[1]) - 1], int(current_object[0])
        )

    # start squeezing an object
    samples = list()
    t = 0

    for _ in range(START_STEP):
        if args.snapshot:
            if t % 500 == 0:
                img_array = env.viewer._read_pixels_as_in_window(resolution=(500, 300))
                img = Image.fromarray(img_array)
                img.save(imgname.format(t))

        readings, contact = env.step_touch(args.dataset, ep)
        if args.mask_contact and not contact:
            readings = np.zeros_like(readings)
        if readings is not None:
            samples.append(readings)
        t += 1

    for i in range(MAX_ITER_PER_EP):
        if args.snapshot:
            if t % 500 == 0:
                img_array = env.viewer._read_pixels_as_in_window(resolution=(500, 300))
                img = Image.fromarray(img_array)
                img.save(imgname.format(t))
        else:
            env.render()

        if i < BEFORE_TOUCH:
            env.move(i, BEFORE_TOUCH)
        elif i >= BEFORE_TOUCH:
            env.move(i, OPEN_CLOSE_DIV)

        readings = env.step(args.dataset, ep)
        if readings is not None:
            samples.append(readings)
        t += 1

    samples = np.array(samples)

    # Add noises
    if args.noise_pos > 0:
        samples[:, 0:11] += np.random.normal(0, np.sqrt(args.noise_pos))
    if args.noise_vel > 0:
        samples[:, 11:22] += np.random.normal(0, np.sqrt(args.noise_vel))

    data = samples
    stiffness = current_object[0]
    shape = current_object[1]
    pos = current_object[2:]

    return data, stiffness, shape, pos


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sim-step", type=int, default=1)
    parser.add_argument("--timestep", type=int, default=0.001)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--mask-contact", type=bool, default=False)
    parser.add_argument("--sim-start", type=int, default=1)
    parser.add_argument("--dataset", type=int, default=0)
    # train:0, train_big:1, test:2, test_big:3, train_full:4
    parser.add_argument("--data", type=str, default="dataset")
    parser.add_argument("--data-name", type=str, default="data")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--noise-pos", type=float, default=0)
    parser.add_argument("--noise-vel", type=float, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    args, _ = parser.parse_known_args()

    main(args)
