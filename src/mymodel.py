import torch
import torch.nn as nn


class LSTM_Variance(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_dim,
        n_layers,
        stiffness_outputs: int = None,
        shape_outputs: int = None,
    ):
        super(LSTM_Variance, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.stiffness_output = stiffness_outputs
        self.shape_output = shape_outputs
        self.mu_output = stiffness_outputs
        self.sigma_output = 1

        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.fc_1 = nn.Linear(hidden_dim, stiffness_outputs)
        self.fc_2 = nn.Linear(hidden_dim, shape_outputs)
        self.fc_4 = nn.Linear(hidden_dim, self.sigma_output)

    def forward(self, x):
        h_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        c_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = self.dropout(out)
        batchsize, length, _ = out.shape
        out = torch.reshape(out, (batchsize * length, self.hidden_dim))
        y = self.fc_1(out)
        z = self.fc_2(out)
        sigma = self.fc_4(out)
        y = torch.reshape(y, (batchsize, length, self.stiffness_output))
        z = torch.reshape(z, (batchsize, length, self.shape_output))
        sigma = torch.reshape(sigma, (batchsize, length, self.sigma_output))

        return y, z, sigma
