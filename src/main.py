import os

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# own modules
from src.models import LSTMForecast
from src.utils import (
    get_dataloaders,
    save_model,
    set_seed,
    RegressionMetrics,
    plot_logs,
    plot_prediction,
    load_model,
    generate_forecast_video
)

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)
torch.set_num_threads(8)

CSV_DATA_PATH: str = "data/"
MODEL_NAME: str = "lstm_baseline"
LOG_DIR: str = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
PLOT_DIR: str = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Model & training hyperparameters
epochs: int = 1
learning_rate: float = 1e-2
batch_size: int = 16
history: int = 30
horizon: int = 10
spatial_stride: int = 3
hidden_size: int = 512
num_layers: int = 2
dropout: float = 0.5
patience: int = 2
factor: float = 0.5

# Logging containers
train_logs: dict[str, list[float]] = {"loss": [], "mse": [], "r2": []}
val_logs: dict[str, list[float]] = {"loss": [], "mse": [], "r2": []}


def main() -> None:
    global train_logs, val_logs

    train_loader, val_loader, test_loader, simulation = get_dataloaders(
        csvs_path=CSV_DATA_PATH,
        batch_size=batch_size,
        history=history,
        horizon=horizon,
        spatial_stride=spatial_stride
    )

    global_mean: float = float(np.load(os.path.join(LOG_DIR, "global_mean.npy")))
    global_std: float = float(np.load(os.path.join(LOG_DIR, "global_std.npy")))

    input_size: int = next(iter(train_loader))[0].shape[-1]

    model: LSTMForecast = LSTMForecast(
        input_size=input_size,
        hidden_size=hidden_size,
        dropout=dropout,
        num_layers=num_layers,
    ).to(device)

    criterion: torch.nn.Module = torch.nn.MSELoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                                                                       patience=patience, factor=factor)

    for epoch in tqdm(range(epochs)):
        train_loss: float = train_step(model, train_loader, criterion, optimizer, epoch)
        val_loss: float = val_step(model, val_loader, criterion, epoch, global_mean, global_std)
        scheduler.step(val_loss)

    save_model(model, MODEL_NAME)
    plot_logs(train_logs, val_logs, LOG_DIR)
    results: dict[str, float] = test_step(model, test_loader, device)

    print(f"[Test Results] | MSE: {results['mse']:.4f} | MAE: {results['mae']:.4f} | R²: {results['r2']:.4f}")

    # Generate video
    generate_forecast_video(
        model=model,
        sim_data=simulation,
        history=30,
        horizon=10,
        global_mean=global_mean,
        global_std=global_std,
        save_path="plots/forecast_video.gif",
    )


def train_step(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int
) -> float:
    model.train()
    losses: list[float] = []
    metrics: RegressionMetrics = RegressionMetrics()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_hat: torch.Tensor = model(x)
        loss: torch.Tensor = loss_fn(y_hat, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        metrics.update(y_hat, y)
        losses.append(loss.item())

    avg_loss: float = float(np.mean(losses))
    results: dict[str, float] = metrics.compute()

    train_logs["loss"].append(avg_loss)
    train_logs["mse"].append(results["mse"])
    train_logs["r2"].append(results["r2"])

    print(f"[Train] Epoch {epoch + 1} | Loss: {avg_loss:.4f} | MSE: {results['mse']:.4f} | MAE: {results['mae']:.4f} | R²: {results['r2']:.4f}")
    return avg_loss


def val_step(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    epoch: int,
    global_mean: float,
    global_std: float
) -> float:
    model.eval()
    losses: list[float] = []
    metrics: RegressionMetrics = RegressionMetrics()

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            y_hat: torch.Tensor = model(x)
            loss: torch.Tensor = loss_fn(y_hat, y)

            metrics.update(y_hat, y)
            losses.append(loss.item())

            # if i % 5 == 0:
            #     plot_prediction(
            #         y_true=y[0],
            #         y_pred=y_hat[0],
            #         epoch=epoch,
            #         plot_dir=PLOT_DIR,
            #         batch_idx=i,
            #         global_mean=global_mean,
            #         global_std=global_std,
            #         original_input=x[0],
            #     )

    avg_loss: float = float(np.mean(losses))
    results: dict[str, float] = metrics.compute()

    val_logs["loss"].append(avg_loss)
    val_logs["mse"].append(results["mse"])
    val_logs["r2"].append(results["r2"])

    print(f"[Val]   Epoch {epoch + 1} | Loss: {avg_loss:.4f} | MSE: {results['mse']:.4f} | MAE: {results['mae']:.4f} | R²: {results['r2']:.4f}")
    return avg_loss


def test_step(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device
) -> dict[str, float]:
    model.eval()
    metrics: RegressionMetrics = RegressionMetrics()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            y_hat: torch.Tensor = model(x)
            metrics.update(y_hat, y)

    return metrics.compute()


if __name__ == "__main__":
    main()
