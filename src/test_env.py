from argparse import ArgumentParser
import os
from PIL import Image
from myenv import HandEnv

NUM_EPISODES = 1
MAX_ITER_PER_EP = 2000
OPEN_CLOSE_DIV = 1600
START_STEP = 200
BEFORE_TOUCH = 200


def log_into_file(args):
    current_env = 0

    env_spec = HandEnv.get_std_spec(args)
    env = HandEnv(**env_spec)

    current_env = 1
    episode = 1
    env.load_env(current_env - 1)
    current_object = env.reset(0, episode)

    print("Stiffness: {0[0]:5.1f}, Shape: {0[1]:.0f}".format(current_object))

    env.viewer.cam.distance = 0.35
    env.viewer.cam.lookat[0] = 0.1
    env.viewer.cam.lookat[1] = -0.025
    env.viewer.cam.lookat[2] = 0.02
    env.viewer.cam.azimuth = 135
    env.viewer.cam.elevation = -20

    # env.render()
    env.step_touch(args.dataset, episode)
    img_array = env.viewer._read_pixels_as_in_window()
    img = Image.fromarray(img_array)

    os.makedirs("snapshots", exist_ok=True)
    img.save("snapshots/env.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sim-step", type=int, default=1)
    parser.add_argument("--timestep", type=int, default=0.001)
    parser.add_argument("--no-vis", dest="vis", action="store_false")
    parser.add_argument("--mask-contact", type=bool, default=False)
    parser.add_argument("--sim-start", type=int, default=1)
    parser.add_argument("--dataset", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--noise-pos", type=float, default=0)
    parser.add_argument("--noise-vel", type=float, default=0)
    args, _ = parser.parse_known_args()
    log_into_file(args)
