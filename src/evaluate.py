from dataclasses import dataclass, field
import enum
import json
import os
import pickle
import numpy as np
import torch
import tyro
from utils import CustomDataset, norm
import mymodel


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
    method: MethodType = MethodType.unspecified


def main(modelname: str, dataname: str, outdir: str, /, use_gpu: bool = False):
    # load json data
    f_in = open(os.path.join(os.path.dirname(__file__), "../config/config.json"), "r")
    settings = json.load(f_in)

    # load json parameters
    seed: int = settings["seed"]
    sequence_length: int = settings["sequence_length"]
    ep_size: int = settings["ep_size"]
    stiffness_range: float = settings["stiffness_range"]
    stiffness_min: float = settings["stiffness_min"]
    input_size: int = settings["input_size"]

    torch.manual_seed(seed)

    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    net = torch.load(modelname, map_location=device)
    net.eval()

    sequence_length = net.input_size // input_size

    dataset = load_dataset(dataname, sequence_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=ep_size)

    y_list = []
    z_list = []
    y_raw_list = []
    ln_var_list = []
    z_raw_list = []

    with torch.no_grad():
        for x, y, z in dataloader:
            x = x.to(device).reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
            y = y.to(device)
            z = z.to(device).type(torch.long) - 1

            y_raw, z_raw, ln_var = net(x)

            y_list.append(y.detach().cpu())
            z_list.append(z.detach().cpu())
            y_raw_list.append(y_raw.detach().cpu())
            ln_var_list.append(ln_var.detach().cpu())
            z_raw_list.append(z_raw.detach().cpu())

    y_array = torch.concat(y_list, dim=0).numpy()
    z_array = torch.concat(z_list, dim=0).numpy()
    mu_raw_tensor = torch.concat(y_raw_list, dim=0)
    ln_var_tensor = torch.concat(ln_var_list, dim=0)
    z_raw_tensor = torch.concat(z_raw_list, dim=0)

    method = MethodType.unspecified
    if isinstance(net, mymodel.LSTM_Variance):
        method = MethodType.proposed
    elif isinstance(net, mymodel.LSTM_Baseline):
        method = MethodType.baseline
    else:
        method = MethodType.unspecified

    result = EvaluationResult(
        stiffness_true=y_array,
        shape_true=z_array,
        stiffness_mu=norm(mu_raw_tensor.numpy(), stiffness_range, stiffness_min),
        stiffness_sigma=np.exp(ln_var_tensor.numpy() * 0.5),
        shape_est=torch.softmax(z_raw_tensor, dim=2).numpy(),
        n_data=y_array.shape[0],
        sequence_length=sequence_length,
        method=method,
    )

    with open(outdir, "wb") as fp:
        pickle.dump(result, fp)


def load_dataset(path, sequence_size):
    with open(path, "rb") as fp:
        raw_data = pickle.load(fp)
    dataset = CustomDataset(raw_data, sequence_size, augment=False)
    return dataset


if __name__ == "__main__":
    tyro.cli(main)
