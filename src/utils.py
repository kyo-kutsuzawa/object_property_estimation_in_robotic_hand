import json
import numpy as np
from torch.utils.data import Dataset

# load json data
f_in = open("config/config.json", "r")
settings = json.load(f_in)

# load json parameters
seed = settings["seed"]
input_size = settings["input_size"]
sequence_length = settings["sequence_length"]
num_steps = settings["num_steps"]
hidden_dim = settings["hidden_dim"]
n_layers = settings["n_layers"]
num_stiffness_outputs = settings["num_stiffness_outputs"]
num_shape_outputs = settings["num_shape_outputs"]
ep_size = settings["ep_size"]
batch_size = settings["batch_size"]
num_episodes_train = settings["num_episodes_train"]
num_episodes_test = settings["num_episodes_test"]
epochs = settings["epochs"]
lr = settings["lr"]
gamma = settings["gamma"]
clip_grad_norm = settings["clip_grad_norm"]
stiffness_range = settings["stiffness_range"]
stiffness_min = settings["stiffness_min"]
weight_decay = settings["weight_decay"]
tensorboard_path = settings["tensorboard_path"]
num_splits = settings["num_splits"]
alpha = settings["alpha"]
beta = settings["beta"]
g = settings["g"]
model_save_interval = settings["model_save_interval"]

norm = lambda raw_output, y_range, y_min: y_range * (raw_output + 1) * 0.5


class CustomDataset(Dataset):
    def __init__(self, data, sequence_size, augment):
        self.random_seed = 42
        self.data = np.array(data["data"])
        self.sequence_size = sequence_size
        self.labels_stiffness = np.array(data["stiffness"])
        self.labels_shape = np.array(data["shape"])

        if augment:
            self.data[:, :, 0:22] += np.random.normal(
                0.0, 0.05, [*self.data.shape[:2], 22]
            )

        self.num_episodes = len(self.data)
        self.episode_length = len(self.data[0])
        self.total_steps = self.num_episodes * self.episode_length

        self.seq_data, self.labels_stiffness, self.labels_shape = (
            self.make_sequence_data()
        )

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        return self.seq_data[idx], self.labels_stiffness[idx], self.labels_shape[idx]

    def __add__(self, other):
        self.seq_data = np.concatenate([self.seq_data, other.data], 0)
        self.labels_stiffness = np.concatenate(
            [self.labels_stiffness, other.labels_stiffness], 0
        )
        self.labels_shape = np.concatenate([self.labels_shape, other.labels_shape], 0)
        return self

    def make_sequence_data(self):
        num_episodes, num_steps, n_dim = self.data.shape
        num_train_steps = num_steps // self.sequence_size

        seq_data_all_episodes = np.zeros(
            (num_episodes, num_train_steps, self.sequence_size, n_dim), dtype=np.float32
        )
        labels_stiffness_all_episodes = np.zeros(
            (num_episodes, num_train_steps, 1), dtype=np.float32
        )
        labels_shape_all_episodes = np.zeros(
            (num_episodes, num_train_steps, 1), dtype=np.float32
        )

        for episode in range(num_episodes):
            for step in range(num_train_steps):
                seq_data_all_episodes[episode][step] = self.data[
                    episode, step * self.sequence_size : (step + 1) * self.sequence_size
                ]
                labels_stiffness_all_episodes[episode][step] = self.labels_stiffness[
                    episode
                ]
                labels_shape_all_episodes[episode][step] = self.labels_shape[episode]

        return (
            seq_data_all_episodes,
            labels_stiffness_all_episodes,
            labels_shape_all_episodes,
        )
