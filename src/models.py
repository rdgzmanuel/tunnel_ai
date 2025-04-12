import torch
import torch.nn as nn


class LSTMForecast(nn.Module):
    """
    Baseline LSTM model for tunnel temperature prediction.
    """
    def __init__(self, input_size: int, hidden_size: int, dropout: float, num_layers: int, output_horizon: int) -> None:
        """
        Args:
            input_size (int): Number of spatial points used as input features.
            hidden_size (int): Number of hidden units in LSTM.
            num_layers (int): Number of stacked LSTM layers.
            output_horizon (int): Number of future time steps to predict.
        """
        super().__init__()

        self.lstm: nn.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                     num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc: nn.Linear = nn.Linear(hidden_size, input_size * output_horizon)
        self.output_horizon: int = output_horizon
        self.input_size: int = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape [batch, time, spatial_points]

        Returns:
            Tensor: Output tensor of shape [batch, horizon, spatial_points]
        """
        _, (hn, _) = self.lstm(x)  # Take hidden state from last layer
        hn_last: torch.Tensor = hn[-1]  # shape: [batch, hidden_size]
        out: torch.Tensor = self.fc(hn_last)  # shape: [batch, horizon * spatial_points]
        out = out.view(-1, self.output_horizon, self.input_size)
        return out


class TCNBlock(nn.Module):
    """
    A single block for Temporal Convolutional Network (TCN).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.conv: nn.Conv1d = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation, dilation=dilation
        )
        self.relu: nn.ReLU = nn.ReLU()
        self.norm: nn.BatchNorm1d = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.conv(x)
        out = out[:, :, :-self.conv.padding[0]]  # remove excess padding
        out = self.relu(out)
        out = self.norm(out)
        return out


class TCNForecast(nn.Module):
    """
    Temporal Convolutional Network for spatiotemporal tunnel temperature prediction.
    """
    def __init__(self, input_size: int, horizon: int, num_layers: int = 4, kernel_size: int = 3, channels: int = 64) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_ch: int = input_size

        for i in range(num_layers):
            layers.append(TCNBlock(
                in_channels=in_ch,
                out_channels=channels,
                kernel_size=kernel_size,
                dilation=2**i
            ))
            in_ch = channels

        self.network: nn.Sequential = nn.Sequential(*layers)
        self.final: nn.Linear = nn.Linear(channels, input_size * horizon)
        self.horizon: int = horizon
        self.input_size: int = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape [batch, time, spatial_points]

        Returns:
            Tensor: Output tensor of shape [batch, horizon, spatial_points]
        """
        x = x.permute(0, 2, 1)  # [batch, spatial_points, time]
        out: torch.Tensor = self.network(x)
        out = out.mean(dim=2)  # global temporal average
        out = self.final(out)
        out = out.view(-1, self.horizon, self.input_size)
        return out
