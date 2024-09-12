import enum
import json
import os
import pickle
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import tyro
import mymodel
from utils import CustomDataset, norm


# load json data
f_in = open(os.path.join(os.path.dirname(__file__), "../config/config.json"), "r")
settings = json.load(f_in)

# load json parameters
seed: int = settings["seed"]
input_size: int = settings["input_size"]
sequence_length: int = settings["sequence_length"]
num_steps: int = settings["num_steps"]
hidden_dim: int = settings["hidden_dim"]
n_layers: int = settings["n_layers"]
num_stiffness_outputs: int = settings["num_stiffness_outputs"]
num_shape_outputs: int = settings["num_shape_outputs"]
ep_size: int = settings["ep_size"]
batch_size: int = settings["batch_size"]
num_episodes_train: int = settings["num_episodes_train"]
num_episodes_test: int = settings["num_episodes_test"]
epochs: int = settings["epochs"]
lr: float = settings["lr"]
gamma: float = settings["gamma"]
clip_grad_norm: float = settings["clip_grad_norm"]
stiffness_range: float = settings["stiffness_range"]
stiffness_min: float = settings["stiffness_min"]
weight_decay: float = settings["weight_decay"]
tensorboard_path: str = settings["tensorboard_path"]
num_splits: int = settings["num_splits"]
alpha: int = settings["alpha"]
beta: int = settings["beta"]
g: int = settings["g"]
model_save_interval: float = settings["model_save_interval"]
dropout_ratio: float = settings["dropout_ratio"]

torch.manual_seed(seed)


class MethodType(enum.Enum):
    baseline = enum.auto()
    proposed = enum.auto()


def main(
    datadir: str = "dataset/train.pickle",
    outdir: str = "result",
    device: str = "cpu",
    augment: bool = False,
    augment_online: bool = False,
    method: MethodType = MethodType.proposed,
    loss_weight: float = 1.0,
    sequence_length: int = settings["sequence_length"],
    lr: float = settings["lr"],
    weight_decay: float = settings["weight_decay"],
    dropout_ratio: float = settings["dropout_ratio"],
    epochs: int = settings["epochs"],
    n_layers: int = settings["n_layers"],
    hidden_dim: int = settings["hidden_dim"],
    model_save_interval: float = settings["model_save_interval"],
):
    _device = torch.device(device)

    datadir = load_dataset(datadir, sequence_length, augment, _device)
    n_train = int(len(datadir.seq_data) * 0.7)
    n_val = int(len(datadir.seq_data) - n_train)
    train_dataset, val_dataset = torch.utils.data.random_split(
        datadir, [n_train, n_val]
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=ep_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=ep_size)

    if method == MethodType.proposed:
        net = mymodel.LSTM_Variance(
            input_size=input_size * sequence_length,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout_ratio=dropout_ratio,
            stiffness_outputs=num_stiffness_outputs,
            shape_outputs=num_shape_outputs,
        ).to(_device)
    elif method == MethodType.baseline:
        net = mymodel.LSTM_Baseline(
            input_size=input_size * sequence_length,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout_ratio=dropout_ratio,
            stiffness_outputs=num_stiffness_outputs,
            shape_outputs=num_shape_outputs,
        ).to(_device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    writer = SummaryWriter(os.path.join(outdir, str(int(time.time()))))

    for epoch in range(epochs):
        epoch_loss_train = 0.0
        net.train()
        for i, (x, y, z) in enumerate(train_dataloader):
            # Add noises
            if augment_online:
                with torch.no_grad():
                    obs_shape = x[:, :, :, :22].size()
                    std = float(10 ** np.random.uniform(np.log10(1e-4), np.log10(1e-1)))
                    x[:, :, :, :22] += torch.normal(0, std, obs_shape).to(_device)

            x = x.to(_device).reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
            y = y.to(_device)
            z = z.to(_device).type(torch.long) - 1

            y_raw, z_raw, ln_var = net(x)
            y_pred = norm(y_raw, stiffness_range, stiffness_min)

            if method == MethodType.proposed:
                loss = lossfunc_proposed(y, y_pred, ln_var, z, z_raw)
            elif method == MethodType.baseline:
                loss = lossfunc_baseline(y, y_pred, loss_weight, z, z_raw)
            epoch_loss_train += float(loss.detach().cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss_val = 0.0
        net.eval()
        with torch.no_grad():
            for i, (x, y, z) in enumerate(val_dataloader):
                x = x.to(_device).reshape(
                    x.shape[0], x.shape[1], x.shape[2] * x.shape[3]
                )
                y = y.to(_device)
                z = z.to(_device).type(torch.long) - 1

                y_raw, z_raw, ln_var = net(x)
                y_pred = norm(y_raw, stiffness_range, stiffness_min)

                if method == MethodType.proposed:
                    loss = lossfunc_proposed(y, y_pred, ln_var, z, z_raw)
                elif method == MethodType.baseline:
                    loss = lossfunc_baseline(y, y_pred, loss_weight, z, z_raw)
                epoch_loss_val += float(loss.detach().cpu().numpy())

        epoch_loss_train /= len(train_dataloader)
        epoch_loss_val /= len(val_dataloader)
        print(
            f"{epoch} epoch: ",
            f"Train loss = {epoch_loss_train: 8.3f}, ",
            f"Val loss = {epoch_loss_val: 8.3f}",
        )
        writer.add_scalar("trainig/loss", epoch_loss_train, epoch)
        writer.add_scalar("val/loss", epoch_loss_val, epoch)

        if epoch % 10 == 0:
            os.makedirs(outdir, exist_ok=True)
            torch.save(net, os.path.join(outdir, "nn_latest.pth"))

        if epoch % model_save_interval == 0:
            os.makedirs(outdir, exist_ok=True)
            torch.save(net, os.path.join(outdir, f"nn_{epoch}.pth"))

    writer.close()


def load_dataset(path, sequence_size, augment=False, device=torch.device("cpu")):
    with open(path, "rb") as fp:
        raw_data = pickle.load(fp)
    dataset = CustomDataset(raw_data, sequence_size, augment, device)
    return dataset


def lossfunc_proposed(y_true, y_pred, ln_var, z_true, z_pred):
    shape_criterion = torch.nn.CrossEntropyLoss()

    k_diff = y_true - y_pred
    variance_loss = torch.mean(k_diff**2 / torch.exp(ln_var) + ln_var)
    classification_loss = shape_criterion(
        torch.reshape(z_pred, (-1, z_pred.shape[-1])), torch.flatten(z_true)
    )

    total_loss = variance_loss * 0.5 + classification_loss

    return total_loss


def lossfunc_baseline(y_true, y_pred, alpha, z_true, z_pred):
    shape_criterion = torch.nn.CrossEntropyLoss()

    k_diff = y_true - y_pred
    variance_loss = torch.mean(k_diff**2)
    classification_loss = shape_criterion(
        torch.reshape(z_pred, (-1, z_pred.shape[-1])), torch.flatten(z_true)
    )

    total_loss = alpha * variance_loss * 0.5 + classification_loss

    return total_loss


if __name__ == "__main__":
    tyro.cli(main)
