import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# own modules
from src.models import LSTMForecast
from src.utils import (
    get_dataloaders,
    save_model,
    set_seed,
    Accuracy
)


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)
torch.set_num_threads(8)

CSV_DATA_PATH: str = "data/csv"
MODEL_NAME: str = "lstm_baseline"


# Model & training hyperparameters
epochs: int = 25
learning_rate: float = 1e-3
batch_size: int = 64
history: int = 30
horizon: int = 10
spatial_stride: int = 3
hidden_size: int = 128
num_layers: int = 2


def main() -> None:
    train_loader, val_loader, _ = get_dataloaders(
        csv_path=CSV_DATA_PATH,
        batch_size=batch_size,
        history=history,
        horizon=horizon,
        spatial_stride=spatial_stride
    )

    input_size: int = next(iter(train_loader))[0].shape[-1]  # number of spatial points after stride

    model: LSTMForecast = LSTMForecast(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_horizon=horizon
    ).to(device)

    writer: SummaryWriter = SummaryWriter(f"runs/{MODEL_NAME}")

    criterion: torch.nn.Module = torch.nn.MSELoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(epochs)):
        train_loss: float = train_step(model, train_loader, criterion, optimizer, writer, epoch)
        val_loss: float = val_step(model, val_loader, criterion, writer, epoch)

    save_model(model, MODEL_NAME)


def train_step(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    epoch: int
) -> float:
    """
    
    """
    model.train()
    losses: list[float] = []
    accuracies: list[float] = []
    accuracy: Accuracy = Accuracy()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_hat: torch.Tensor = model(x)
        loss: torch.Tensor = loss_fn(y_hat, y)

        accuracy.update(y_hat, y.long())
        accuracy_value: float = accuracy.compute()

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accuracies.append(accuracy_value)

    avg_loss: float = np.mean(losses)
    avg_accuracy: float = np.mean(accuracies)
    writer.add_scalar("train/loss", avg_loss, epoch)
    writer.add_scalar("train/accuracy", avg_accuracy, epoch)
    return avg_loss


def val_step(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    writer: SummaryWriter,
    epoch: int
) -> float:
    model.eval()
    losses: list[float] = []
    accuracies: list[float] = []
    accuracy: Accuracy = Accuracy()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            y_hat: torch.Tensor = model(x)
            loss: torch.Tensor = loss_fn(y_hat, y)

            accuracy.update(y_hat, y.long())
            accuracy_value: float = accuracy.compute()
            losses.append(loss.item())
            accuracies.append(accuracy_value)

    avg_loss: float = np.mean(losses)
    avg_accuracy: float = np.mean(accuracies)
    writer.add_scalar("val/loss", avg_loss, epoch)
    writer.add_scalar("val/accuracy", avg_accuracy, epoch)
    return avg_loss


if __name__ == "__main__":
    main()