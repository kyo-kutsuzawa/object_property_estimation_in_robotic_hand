import glfw
from mujoco_py import (
    load_model_from_path,
    MjSim,
    builder,
    MjViewer,
)
import numpy as np
import os
import random


class Env(object):
    def __init__(self, sim_start, sim_step, num_envs, timestep, dataset):
        self.sim_start = sim_start
        self.sim_step = sim_step
        self.num_envs = num_envs
        self.timestep = timestep
        self.dataset = dataset

    def step(self, *args):
        raise NotImplementedError("Not implemented")

    def reset(self):
        raise NotImplementedError("Not implemented")


class HandEnv(Env):
    joint_ids_cyl = list(range(17, 209))
    joint_ids_box = list(range(17, 185))
    joint_ids_bal = list(range(17, 235))
    tendon_ids = list(range(1))
    _assetdir = os.path.join(os.path.dirname(__file__), "..", "assets")
    cylinder = os.path.join(_assetdir, "xml_new/soft_cylinder.xml")
    box = os.path.join(_assetdir, "xml_new/soft_box.xml")
    ball = os.path.join(_assetdir, "xml_new/soft_ball.xml")
    cylinder_big = os.path.join(_assetdir, "xml_new/soft_cylinder_big.xml")
    box_big = os.path.join(_assetdir, "xml_new/soft_box_big.xml")
    ball_big = os.path.join(_assetdir, "xml_new/soft_ball_big.xml")
    cylinder_small = os.path.join(_assetdir, "xml_new/soft_cylinder_small.xml")
    box_small = os.path.join(_assetdir, "xml_new/soft_box_small.xml")
    ball_small = os.path.join(_assetdir, "xml_new/soft_ball_small.xml")
    cylinder_pos = os.path.join(_assetdir, "xml_new/soft_cylinder_pos.xml")
    box_pos = os.path.join(_assetdir, "xml_new/soft_box_pos.xml")
    ball_pos = os.path.join(_assetdir, "xml_new/soft_ball_pos.xml")

    # test
    cylinder_middle = os.path.join(_assetdir, "xml_new/soft_cylinder_middle.xml")
    box_middle = os.path.join(_assetdir, "xml_new/soft_box_middle.xml")
    ball_middle = os.path.join(_assetdir, "xml_new/soft_ball_middle.xml")
    cylinder_pos_test = os.path.join(_assetdir, "xml_new/soft_cylinder_pos_test.xml")
    box_pos_test = os.path.join(_assetdir, "xml_new/soft_box_pos_test.xml")
    ball_pos_test = os.path.join(_assetdir, "xml_new/soft_ball_pos_test.xml")

    only_hand = os.path.join(_assetdir, "xml_new/only_hand.xml")

    # list of bodies that are check for collision (partial names are enough)
    finger_names = ["g11", "g12", "g13", "g2"]
    obj_name = "OBJ"

    ff_pos, rf_pos, th_pos = list(), list(), list()

    def __init__(
        self, sim_start, sim_step, num_envs, timestep, dataset, model=box, is_vis=True
    ):
        super().__init__(sim_start, sim_step, num_envs, timestep, dataset)

        # setup environment and viewer
        self.num_envs = num_envs
        self.dataset = dataset
        self.is_vis = is_vis
        self.model = model
        models = [self.cylinder, self.box, self.ball]
        self.models = models
        selected_model = random.choice(models)
        self.random_model = selected_model

        if self.dataset in [0, 1, 2, 3, 4]:
            scene = load_model_from_path(self.model)  # train
        elif self.dataset in [5, 6]:
            scene = load_model_from_path(self.random_model)  # test
        if self.dataset in [20, 21, 22, 23, 24]:
            scene = load_model_from_path(self.model)  # train
        elif self.dataset == 100:
            scene = load_model_from_path(self.model)  # for paper

        self.sim = MjSim(scene)
        if self.is_vis:
            self.viewer = MjViewer(self.sim)
        self.process_flag = False
        self.is_closing = True
        self.sim.model.opt.timestep = timestep

        self.images = []
        self.time = 0
        self.width = 1920
        self.height = 1080

    def load_env(self, num):
        self.num = num

        # Training data (normal size)
        if self.dataset == 0:
            if self.num < 200:
                self.model = self.cylinder
            elif 200 <= self.num < 400:
                self.model = self.box
            elif 400 <= self.num < 600:
                self.model = self.ball

        # Training data (bigger size)
        if self.dataset == 1:
            if self.num < 200:
                self.model = self.cylinder_big
            elif 200 <= self.num < 400:
                self.model = self.box_big
            elif 400 <= self.num < 600:
                self.model = self.ball_big

        # Test data (normal size)
        if self.dataset == 2:
            if self.num < 7:
                self.model = self.cylinder
            elif 7 <= self.num < 14:
                self.model = self.box
            elif 14 <= self.num < 21:
                self.model = self.ball

        # Test data (bigger size)
        if self.dataset == 3:
            if self.num < 7:
                self.model = self.cylinder_big
            elif 7 <= self.num < 14:
                self.model = self.box_big
            elif 14 <= self.num < 21:
                self.model = self.ball_big

        # Training data (full)
        if self.dataset == 4:
            if self.num < 200:
                self.model = self.cylinder
            elif self.num < 400:
                self.model = self.box
            elif self.num < 600:
                self.model = self.ball
            elif self.num < 800:
                self.model = self.cylinder_big
            elif self.num < 1000:
                self.model = self.box_big
            elif self.num < 1200:
                self.model = self.ball_big
            elif self.num < 1400:
                self.model = self.cylinder_small
            elif self.num < 1600:
                self.model = self.box_small
            elif self.num < 1800:
                self.model = self.ball_small

        # Only hand (used for a figure in a paper)
        if self.dataset == 100:
            self.model = self.only_hand

        scene = load_model_from_path(self.model)
        self.sim = MjSim(scene)
        self.sim.model.opt.timestep = self.timestep
        self.process_flag = False

        if self.is_vis:
            if self.viewer is not None:
                glfw.destroy_window(self.viewer.window)
            self.viewer = MjViewer(self.sim)

    def load_env2(self):
        scene = load_model_from_path(self.model)
        self.sim = MjSim(scene)
        self.sim.model.opt.timestep = self.timestep
        self.process_flag = False

        if self.is_vis:
            if self.viewer is not None:
                glfw.destroy_window(self.viewer.window)
            self.viewer = MjViewer(self.sim)

    def load_env_random(self, num):
        self.num = num
        if self.dataset == 5:
            selected_model = random.choice(self.models)

        self.random_model = selected_model
        if self.num < 1000000:
            scene = load_model_from_path(self.random_model)
            self.sim = MjSim(scene)
            self.sim.model.opt.timestep = self.timestep
            self.process_flag = False
        else:
            print("Wrong number,")

    def load_env_movie(self, num):
        self.num = num
        if self.num < 1000000:
            scene = load_model_from_path(self.cylinder)
            self.sim = MjSim(scene)
            self.sim.model.opt.timestep = self.timestep
            self.process_flag = False
        else:
            print("Wrong number,")

    def step(self, dataset, episode, num_steps=-1, actions=None):
        if num_steps < 1:
            num_steps = self.sim_step
        try:
            num_steps = int(num_steps)
            for _ in range(num_steps):
                self.sim.step()
        except builder.MujocoException:
            self.reset(dataset, episode)

        return np.asanyarray(self.sensor_sensordata(), dtype=float).reshape(
            -1
        )  # touch sensor off

    def step_touch(self, dataset, episode, num_steps=-1, actions=None, min_dist=0.1):
        if num_steps < 1:
            num_steps = self.sim_step
        try:
            num_steps = int(num_steps)
            for _ in range(num_steps):
                self.sim.step()
        except builder.MujocoException:
            self.reset(dataset, episode)

        return np.asanyarray(self.get_sensor_sensordata(), dtype=object).reshape(
            -1
        )  # touch sensor on

    def capture(self):
        self.render_off.render(self.width, self.height, camera_id=0)
        image = self.render_off.read_pixels(self.width, self.height, depth=False)
        image = np.flipud(image)
        self.images.append(image)

    def save_video(self):
        print(len(self.images))
        import imageio

        codec_name = "libx264"
        imageio.mimsave("video.mp4", self.images, fps=50, codec=codec_name)
        self.images = []

    def reset(self, dataset, episode):  # train_dataset
        if dataset == 5:
            current_stiffness = self.set_new_stiffness()
        elif dataset in [0, 1, 4]:
            current_stiffness = self.set_new_stiffness_4(episode)
        elif dataset in [2, 3]:
            current_stiffness = self.set_new_stiffness_128(episode)

        if dataset in [0, 1, 2, 3, 4]:
            obj_pos = self.set_object_position()

        self.sim.reset()
        self.sim.forward()

        if self.sim_start > 0:
            self.step(dataset, episode, self.sim_start)

        shape = self.model
        if shape in [
            self.cylinder,
            self.cylinder_big,
            self.cylinder_small,
            self.cylinder_pos,
            self.cylinder_middle,
            self.cylinder_pos_test,
        ]:
            current_shape = 1  # cylinder
        elif shape in [
            self.box,
            self.box_big,
            self.box_small,
            self.box_pos,
            self.box_middle,
            self.box_pos_test,
        ]:
            current_shape = 2  # box
        elif shape in [
            self.ball,
            self.ball_big,
            self.ball_small,
            self.ball_pos,
            self.ball_middle,
            self.ball_pos_test,
        ]:
            current_shape = 3  # ball
        elif shape == self.only_hand:
            current_shape = 3  # ball

        labels = [current_stiffness, current_shape] + obj_pos.flatten().tolist()

        return np.asanyarray(labels, dtype=float).reshape(-1)

    def reset2(self, fluctuation=False, pos=None, quat=None):
        if fluctuation:
            obj_pos = self.set_object_position(pos, quat)

        self.sim.reset()
        self.sim.forward()

        if self.sim_start > 0:
            dataset = None
            episode = None
            self.step(dataset, episode, self.sim_start)

        return obj_pos.flatten()

    def reset_random(self):
        current_stiffness = self.set_new_stiffness_random_shape()
        self.sim.reset()
        self.sim.forward()

        if self.sim_start > 0:
            self.step(self.sim_start)
        if self.dataset == 5:
            shape = self.random_model

        print(shape)
        if shape == self.cylinder:
            current_shape = 1  # cylinder
        elif shape == self.box:
            current_shape = 2  # box
        elif shape == self.ball:
            current_shape = 3  # ball

        labels = [current_stiffness, current_shape]

        return np.asanyarray(labels, dtype=float).reshape(-1)

    def reset_movie(self, j):
        current_stiffness = self.set_new_stiffness_movie(j)
        self.sim.reset()
        self.sim.forward()

        if self.sim_start > 0:
            self.step_movie(self.sim_start)

        shape = self.model
        if shape == self.cylinder:
            current_shape = 1  # cylinder
        elif shape == self.box:
            current_shape = 2  # box
        elif shape == self.ball:
            current_shape = 3  # ball

        labels = [current_stiffness, current_shape]

        return np.asanyarray(labels, dtype=float).reshape(-1)

    def get_ob(self):
        ob = np.concatenate(
            [
                self.sim.data.qpos[2].flat,
                self.sim.data.qpos[3].flat,
                self.sim.data.qpos[4].flat,
                self.sim.data.qpos[6].flat,
                self.sim.data.qpos[7].flat,
                self.sim.data.qpos[8].flat,
                self.sim.data.qpos[10].flat,
                self.sim.data.qpos[11].flat,
                self.sim.data.qpos[12].flat,
                self.sim.data.qpos[15].flat,
                self.sim.data.qpos[16].flat,
                self.sim.data.qvel[2].flat,
                self.sim.data.qvel[3].flat,
                self.sim.data.qvel[4].flat,
                self.sim.data.qvel[6].flat,
                self.sim.data.qvel[7].flat,
                self.sim.data.qvel[8].flat,
                self.sim.data.qvel[10].flat,
                self.sim.data.qvel[11].flat,
                self.sim.data.qvel[12].flat,
                self.sim.data.qvel[15].flat,
                self.sim.data.qvel[16].flat,
                self.sim.data.ctrl[1].flat,
                self.sim.data.ctrl[2].flat,
                self.sim.data.ctrl[3].flat,
                self.sim.data.ctrl[5].flat,
                self.sim.data.ctrl[6].flat,
                self.sim.data.ctrl[7].flat,
                self.sim.data.ctrl[9].flat,
                self.sim.data.ctrl[10].flat,
                self.sim.data.ctrl[11].flat,
                self.sim.data.ctrl[14].flat,
                self.sim.data.ctrl[15].flat,
            ]
        )

        return ob

    def sensor_sensordata(self):

        return np.copy(self.get_ob())

    def get_sensor_sensordata(self):
        data = self.sim.data

        # Return true when all fingers can contact an object's body
        is_contact_between_fingers_and_object = False
        fingers_left = self.finger_names
        for coni in range(data.ncon):
            contact = data.contact[coni]
            body1_name = self.sim.model.geom_id2name(contact.geom1)
            body2_name = self.sim.model.geom_id2name(contact.geom2)
            if body1_name is not None and body2_name is not None:
                if self.obj_name in body1_name or self.obj_name in body2_name:
                    for finger_name in fingers_left:
                        is_finger_contact = bool(
                            finger_name in body1_name or finger_name in body2_name
                        )
                        if is_finger_contact:
                            fingers_left.remove(finger_name)
            if len(fingers_left) == 0:
                is_contact_between_fingers_and_object = True
                break

        return np.copy(self.get_ob()), is_contact_between_fingers_and_object

    def toggle_grip(self, i):
        if self.is_closing:
            self.loose_hand(i)
        else:
            self.close_hand(i)

    def move(self, i, num_steps):
        if i <= num_steps:
            t = i * self.sim.model.opt.timestep
            sin_value = np.sin(t)
            if num_steps != 200:
                self.sim.data.ctrl[1] = 0.7 * sin_value * 1.2
                self.sim.data.ctrl[2] = 0.5 * sin_value * 1.2
                self.sim.data.ctrl[3] = 0.25 * sin_value * 1.2
                self.sim.data.ctrl[5] = 0.7 * sin_value * 1.2
                self.sim.data.ctrl[6] = 0.5 * sin_value * 1.2
                self.sim.data.ctrl[7] = 0.25 * sin_value * 1.2
                self.sim.data.ctrl[9] = 0.7 * sin_value * 1.2
                self.sim.data.ctrl[10] = 0.5 * sin_value * 1.2
                self.sim.data.ctrl[11] = 0.25 * sin_value * 1.2
                self.sim.data.ctrl[14] = 0.7 * sin_value * 1.2
                self.sim.data.ctrl[15] = 0.2 * sin_value * 1.2

            elif num_steps == 200:
                self.sim.data.ctrl[1] = 0.7 * 0.25
                self.sim.data.ctrl[2] = 0.5 * 0.25
                self.sim.data.ctrl[3] = 0.25 * 0.25
                self.sim.data.ctrl[5] = 0.7 * 0.25
                self.sim.data.ctrl[6] = 0.5 * 0.25
                self.sim.data.ctrl[7] = 0.25 * 0.25
                self.sim.data.ctrl[9] = 0.7 * 0.25
                self.sim.data.ctrl[10] = 0.5 * 0.25
                self.sim.data.ctrl[11] = 0.25 * 0.25
                self.sim.data.ctrl[14] = 0.7 * 0.25
                self.sim.data.ctrl[15] = 0.2 * 0.25

            self.process_flag = True
        else:
            t = (i - num_steps) * self.sim.model.opt.timestep
            sin_value = np.sin(t)

            self.sim.data.ctrl[0] = 0
            self.sim.data.ctrl[1] = -np.sin(t) * 0.02
            self.sim.data.ctrl[2] = -np.sin(t) * 0.05
            self.sim.data.ctrl[3] = 0
            self.sim.data.ctrl[4] = 0
            self.sim.data.ctrl[5] = -np.sin(t) * 0.02
            self.sim.data.ctrl[6] = -np.sin(t) * 0.05
            self.sim.data.ctrl[7] = 0
            self.sim.data.ctrl[8] = 0
            self.sim.data.ctrl[9] = -np.sin(t) * 0.02
            self.sim.data.ctrl[10] = -np.sin(t) * 0.05
            self.sim.data.ctrl[11] = 0
            self.sim.data.ctrl[12] = 0
            self.sim.data.ctrl[13] = 0
            self.sim.data.ctrl[14] = -np.sin(t) * 0.02
            self.sim.data.ctrl[15] = -np.sin(t) * 0.05

    def close_hand(self, i):
        self.is_closing = True

    def loose_hand(self, i):
        t = i * self.sim.model.opt.timestep

        self.sim.data.ctrl[0] = 0
        self.sim.data.ctrl[1] = -np.sin(t)
        self.sim.data.ctrl[2] = -np.sin(t) * 1.2
        self.sim.data.ctrl[3] = 0
        self.sim.data.ctrl[4] = 0
        self.sim.data.ctrl[5] = -np.sin(t)
        self.sim.data.ctrl[6] = -np.sin(t) * 1.2
        self.sim.data.ctrl[7] = 0
        self.sim.data.ctrl[8] = 0
        self.sim.data.ctrl[9] = -np.sin(t)
        self.sim.data.ctrl[10] = -np.sin(t) * 1.2
        self.sim.data.ctrl[11] = 0
        self.sim.data.ctrl[12] = 0
        self.sim.data.ctrl[13] = 0
        self.sim.data.ctrl[14] = -np.sin(t) * 0.6
        self.sim.data.ctrl[15] = -np.sin(t) * 0.8

        self.process_flag = True

        self.is_closing = False

    def set_new_stiffness(self, range_min=0, range_max=800):
        new_value = np.random.uniform(range_min, range_max)
        shape = self.model

        if shape == self.cylinder or shape == self.cylinder_big:
            self.joint_ids = self.joint_ids_cyl
        elif shape == self.box:
            self.joint_ids = self.joint_ids_box
        elif shape == self.ball:
            self.joint_ids = self.joint_ids_bal

        for i in self.joint_ids:
            self.sim.model.jnt_stiffness[i] = new_value
        for i in self.tendon_ids:
            self.sim.model.tendon_stiffness[i] = new_value

        return new_value

    def set_new_stiffness_4(self, episode, range_min=1, range_max=801):
        episode = (episode - 1) % 200 + 1
        new_value = range_min + 4 * (episode - 1)
        shape = self.model

        if shape in [self.cylinder, self.cylinder_big, self.cylinder_small]:
            self.joint_ids = self.joint_ids_cyl
        elif shape in [self.box, self.box_big, self.box_small]:
            self.joint_ids = self.joint_ids_box
        elif shape in [self.ball, self.ball_big, self.ball_small]:
            self.joint_ids = self.joint_ids_bal
        elif shape == self.only_hand:
            self.joint_ids = self.joint_ids_bal

        for j in self.joint_ids:
            self.sim.model.jnt_stiffness[j] = new_value
        for j in self.tendon_ids:
            self.sim.model.tendon_stiffness[j] = new_value

        return new_value

    def set_new_stiffness_128(self, episode, range_min=3, range_max=801):
        episode = (episode - 1) % 7 + 1
        new_value = range_min + 128 * (episode - 1)
        shape = self.model

        if shape in [self.cylinder, self.cylinder_big, self.cylinder_small]:
            self.joint_ids = self.joint_ids_cyl
        elif shape in [self.box, self.box_big, self.box_small]:
            self.joint_ids = self.joint_ids_box
        elif shape in [self.ball, self.ball_big, self.ball_small]:
            self.joint_ids = self.joint_ids_bal

        for j in self.joint_ids:
            self.sim.model.jnt_stiffness[j] = new_value
        for j in self.tendon_ids:
            self.sim.model.tendon_stiffness[j] = new_value

        return new_value

    def set_new_stiffness2(self, stiffness):
        shape = self.model

        if shape in [self.cylinder, self.cylinder_big, self.cylinder_small]:
            self.joint_ids = self.joint_ids_cyl
        elif shape in [self.box, self.box_big, self.box_small]:
            self.joint_ids = self.joint_ids_box
        elif shape in [self.ball, self.ball_big, self.ball_small]:
            self.joint_ids = self.joint_ids_bal

        for j in self.joint_ids:
            self.sim.model.jnt_stiffness[j] = stiffness
        for j in self.tendon_ids:
            self.sim.model.tendon_stiffness[j] = stiffness

        return stiffness

    def set_new_stiffness_random_shape(self, range_min=1, range_max=800):
        new_value = np.random.uniform(range_min, range_max)

        if self.random_model == self.cylinder or self.random_model == self.cylinder_big:
            self.joint_ids = self.joint_ids_cyl
        elif self.random_model == self.box or self.random_model == self.box_big:
            self.joint_ids = self.joint_ids_box
        elif self.random_model == self.ball or self.random_model == self.ball_big:
            self.joint_ids = self.joint_ids_bal

        for i in self.joint_ids:
            self.sim.model.jnt_stiffness[i] = new_value
        for i in self.tendon_ids:
            self.sim.model.tendon_stiffness[i] = new_value
        return new_value

    def set_new_stiffness_movie(self, j):
        if j == 0:
            new_value = 1
        elif j == 1:
            new_value = 100
        if j == 2:
            new_value = 1000
        shape = self.model

        if shape == self.cylinder:
            self.joint_ids = self.joint_ids_cyl
        elif shape == self.box:
            self.joint_ids = self.joint_ids_box
        elif shape == self.ball:
            self.joint_ids = self.joint_ids_bal
        elif shape == self.cylinder_big:
            self.joint_ids = self.joint_ids_cyl_big
        elif shape == self.box_big:
            self.joint_ids = self.joint_ids_box_big
        elif shape == self.ball_big:
            self.joint_ids = self.joint_ids_bal_big

        for i in self.joint_ids:
            self.sim.model.jnt_stiffness[i] = new_value
        for i in self.tendon_ids:
            self.sim.model.tendon_stiffness[i] = new_value

        return new_value

    def set_object_position(self, pos=None, quat=None):
        if pos is None:
            pos = np.random.uniform(-0.005, 0.005, size=(3,))

        if quat is None:
            theta = np.random.uniform(0, np.deg2rad(5.0))
            d = np.random.normal(0, 1, size=(3,))
            d /= np.linalg.norm(d)

            quat = np.array(
                [
                    np.cos(theta * 0.5),
                    d[0] * np.sin(theta * 0.5),
                    d[1] * np.sin(theta * 0.5),
                    d[2] * np.sin(theta * 0.5),
                ]
            )

        idx = self.sim.model.body_name2id("target_object")
        self.sim.model.body_pos[idx][0:3] += pos[:]
        self.sim.model.body_quat[idx][0:4] = quat[:]

        return np.concatenate([pos, quat], axis=0)

    def get_env(self):
        return self.sim

    def render(self):
        if self.is_vis:
            self.viewer.render()

    @staticmethod
    def get_std_spec(args):
        return {
            "sim_start": args.sim_start,
            "sim_step": args.sim_step,
            "num_envs": args.num_envs,
            "timestep": args.timestep,
            "dataset": args.dataset,
            "is_vis": args.vis,
        }
