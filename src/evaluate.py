import torch
from torch.jit import RecursiveScriptModule
from torch.utils.data import DataLoader
import numpy as np
from typing import Union

from src.utils import (
    get_dataloaders,
    load_model,
    set_seed,
    Accuracy
)
from src.models import LSTMForecast


device: torch.device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
set_seed(42)
torch.set_num_threads(8)

CSV_DATA_PATH: str = "data/csv"
HISTORY: int = 30
HORIZON: int = 10
SPATIAL_STRIDE: int = 3
BATCH_SIZE: int = 64


def main(name: str) -> float:
    """
    Evaluate a trained model on the test set.

    Args:
        name (str): Name of the saved model to load.

    Returns:
        float: accuracy of the model.
    """
    _, _, test_loader = get_dataloaders(
        csv_path=CSV_DATA_PATH,
        batch_size=BATCH_SIZE,
        history=HISTORY,
        horizon=HORIZON,
        spatial_stride=SPATIAL_STRIDE
    )

    model: Union[torch.nn.Module, RecursiveScriptModule] = load_model(name).to(device)
    acc: float = test_step(model, test_loader, device)
    return acc


def test_step(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device
) -> float:
    """
    Run the model on the test set and return average accuracy.

    Args:
        model (torch.nn.Module): Loaded model.
        loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to run inference on.

    Returns:
        float: Accuracy on the test set.
    """
    model.eval()
    metric: Accuracy = Accuracy()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            y_hat: torch.Tensor = model(x)
            y_avg: torch.Tensor = y.mean(dim=1)  # [batch, spatial_points]
            y_hat_avg: torch.Tensor = y_hat.mean(dim=1)  # [batch, spatial_points]

            metric.update(logits=y_hat_avg, labels=y_avg.argmax(dim=1))

    return metric.compute()


if __name__ == "__main__":
    accuracy = main("lstm_baseline")
    print(f"Test Accuracy: {accuracy:.4f}")
