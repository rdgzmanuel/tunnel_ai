import random
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
import io
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from torch.utils.data import Dataset, DataLoader, random_split
from torch.jit import RecursiveScriptModule
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from src.models import LSTMForecast
# T0099.csv


class TunnelTemperatureDataset(Dataset):
    def __init__(
        self,
        csv_folder: str,
        history: int = 30,
        horizon: int = 10,
        spatial_stride: int = 1,
    ) -> None:
        self.csv_folder: str = csv_folder
        self.history: int = history
        self.horizon: int = horizon
        self.spatial_stride: int = spatial_stride

        self.simulations: list[np.ndarray] = []
        self.samples: list[tuple[int, int]] = []

        print(f"Looking for CSV files in: {csv_folder}")
        for root, _, files in os.walk(csv_folder):
            for file in files:
                if file == "T0099.csv":
                    full_path: str = os.path.join(root, file)
                    try:
                        data: np.ndarray = np.loadtxt(full_path, delimiter=",", dtype=np.float32)
                    except Exception as e:
                        print(f"Error reading {full_path}: {e}")
                        continue

                    if data.shape[1] > 1501:
                        data = data[:, :1501]
                    data = data[:, 1:]

                    if data.shape[1] == 1500:
                        downsampled = data[:, ::spatial_stride]
                        self.simulations.append(downsampled)

        if not self.simulations:
            raise RuntimeError("No valid simulations found.")

        self.total_time_steps = self.simulations[0].shape[0]
        self.num_points = self.simulations[0].shape[1]

        for sim_idx, sim in enumerate(self.simulations):
            for t in range(self.total_time_steps - history - horizon):
                self.samples.append((sim_idx, t))

        # Compute global mean and std from all simulations
        all_data = np.concatenate([sim[:self.total_time_steps - horizon] for sim in self.simulations], axis=0)
        self.global_mean: float = float(np.mean(all_data))
        self.global_std: float = float(np.std(all_data))

        print(f"Global mean: {self.global_mean:.4f}, std: {self.global_std:.4f}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sim_idx, t = self.samples[idx]
        sim: np.ndarray = self.simulations[sim_idx]

        x: np.ndarray = sim[t : t + self.history]
        y: np.ndarray = sim[t + self.history + self.horizon - 1]

        x = (x - self.global_mean) / (self.global_std + 1e-6)
        y = (y - self.global_mean) / (self.global_std + 1e-6)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def get_dataloaders(
    csvs_path: str,
    batch_size: int,
    history: int,
    horizon: int,
    spatial_stride: int,
    num_workers: int = 4
) -> tuple[DataLoader, DataLoader, DataLoader]:

    dataset = TunnelTemperatureDataset(
        csv_folder=csvs_path,
        history=history,
        horizon=horizon,
        spatial_stride=spatial_stride,
    )

    # Save computed global stats
    os.makedirs("logs", exist_ok=True)
    np.save("logs/global_mean.npy", dataset.global_mean)
    np.save("logs/global_std.npy", dataset.global_std)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    simulation = dataset.simulations[6]

    return train_loader, val_loader, test_loader, simulation


class Accuracy:
    """
    This class tracks the accuracy of predictions.

    Attributes:
        correct (int): number of correct predictions.
        total (int): total number of examples evaluated.
    """

    def __init__(self) -> None:
        """Initializes correct and total counts to zero."""
        self.correct: int = 0
        self.total: int= 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Updates the count of correct and total predictions.

        Args:
            logits (torch.Tensor): model outputs of shape [batch, num_classes].
            labels (torch.Tensor): ground truth labels of shape [batch].
        """
        predictions: torch.Tensor = logits.argmax(dim=1)
        self.correct += predictions.eq(labels).sum().item()
        self.total += labels.size(0)

    def compute(self) -> float:
        """
        Computes the accuracy.

        Returns:
            float: accuracy as a value between 0 and 1.
        """
        return self.correct / self.total if self.total > 0 else 0.0

    def reset(self) -> None:
        """Resets the correct and total counts to zero."""
        self.correct = 0
        self.total = 0


def load_model(name: str) -> RecursiveScriptModule:
    """
    This function is to load a model from the 'models' folder.

    Args:
        name (str): name of the model to load.

    Returns:
        RecursiveScriptModule: model in torchscript.
    """
    model_path: str = f"/{name}.pt"

    if not os.path.exists(model_path):
        model_path = f"models/{name}.pt"

    model: RecursiveScriptModule = torch.jit.load(model_path)

    return model


def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model in the 'models' folder as a torch.jit.
    It should create the 'models' if it doesn't already exist.

    Args:
        model: pytorch model.
        name: name of the model (without the extension, e.g. name.pt).
    """

    # create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    # save scripted model
    model_scripted: RecursiveScriptModule = torch.jit.script(model.cpu())
    model_scripted.save(f"models/{name}.pt")

    return None


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed: seed number to fix radomness.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None


def load_tensorboard_scalars(model_dir: str, metric: str) -> tuple[list[int], list[float]]:
    """
    Loads scalar data from TensorBoard log files.

    Args:
        model_dir (str): path to the model's log directory.
        metric (str): name of the scalar metric to extract.

    Returns:
        tuple[list[int], list[float]]: steps and corresponding values if metric exists, otherwise None.
    """
    event_acc: EventAccumulator = EventAccumulator(model_dir)
    event_acc.Reload()
    
    if metric not in event_acc.Tags()["scalars"]:
        return None

    events = event_acc.Scalars(metric)
    steps: list[int] = [e.step for e in events]
    values: list[float] = [e.value for e in events]
    
    return steps, values

class RegressionMetrics:
    """
    Tracks MSE, MAE, and R² score for regression tasks.
    """

    def __init__(self) -> None:
        self.y_true: list[float] = []
        self.y_pred: list[float] = []


    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Stores predictions and targets for later evaluation.

        Args:
            preds (torch.Tensor): Predicted values.
            targets (torch.Tensor): Ground truth values.
        """
        self.y_true.extend(targets.detach().cpu().flatten().tolist())
        self.y_pred.extend(preds.detach().cpu().flatten().tolist())

    def compute(self) -> dict[str, float]:
        """
        Computes the regression metrics.

        Returns:
            dict: Dictionary with MSE, MAE, and R².
        """
        mse: float = mean_squared_error(self.y_true, self.y_pred)
        mae: float = mean_absolute_error(self.y_true, self.y_pred)
        r2: float = r2_score(self.y_true, self.y_pred)
        return {"mse": mse, "mae": mae, "r2": r2}

    def reset(self) -> None:
        """Clears stored predictions and labels."""
        self.y_true: list[float] = []
        self.y_pred: list[float] = []


def plot_logs(train_logs, val_logs, log_dir) -> None:
    def _plot(metrics: dict[str, list[float]], title: str, filename: str) -> None:
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
        keys: list[str] = ["loss", "mse", "r2"]

        for i, key in enumerate(keys):
            axs[i].plot(metrics[key], marker="o")
            axs[i].set_title(f"{title} {key.upper()}")
            axs[i].set_xlabel("Epoch")
            axs[i].set_ylabel(key.upper())
            axs[i].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, filename))
        plt.close()

    _plot(train_logs, "Train", "train.png")
    _plot(val_logs, "Validation", "val.png")


def plot_prediction(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    epoch: int,
    plot_dir: str,
    batch_idx: int = 0,
    global_mean: float = 0.0,
    global_std: float = 1.0,
    original_input: torch.Tensor | None = None,
) -> None:
    y_true_np: np.ndarray = y_true.cpu().numpy()
    y_pred_np: np.ndarray = y_pred.cpu().numpy()

    # De-normalize both
    y_true_np = y_true_np * global_std + global_mean
    y_pred_np = y_pred_np * global_std + global_mean

    plt.figure(figsize=(10, 5))
    plt.plot(y_true_np, label="True", linewidth=2)
    plt.plot(y_pred_np, label="Predicted", linestyle="--")
    plt.title(f"Prediction vs Ground Truth (Epoch {epoch + 1}, Batch {batch_idx})")
    plt.xlabel("Spatial Points")
    plt.ylabel("Temperature")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = os.path.join(plot_dir, f"epoch_{epoch + 1}_batch_{batch_idx}.png")
    plt.savefig(filename)
    plt.close()



def generate_forecast_video(
    model: LSTMForecast,
    sim_data: np.ndarray,
    history: int,
    horizon: int,
    global_mean: float,
    global_std: float,
    save_path: str,
    fps: int = 10
) -> None:
    """
    Generate a video showing model forecasts over time.

    Args:
        model (LSTMForecast): Trained model.
        sim_data (np.ndarray): One full simulation [time, spatial_points].
        history (int): Number of past steps to use.
        horizon (int): Forecast horizon.
        global_mean (float): Normalization mean.
        global_std (float): Normalization std.
        save_path (str): Output video path.
        fps (int): Frames per second.
    """
    model.eval()
    frames: list[np.ndarray] = []
    times: range = range(len(sim_data) - history - horizon)
    spatial_points: int = sim_data.shape[1]

    fig, ax = plt.subplots(figsize=(10, 4))

    with torch.no_grad():
        for t in times:
            x_seq: np.ndarray = sim_data[t:t+history]
            x_norm: np.ndarray = (x_seq - global_mean) / (global_std + 1e-6)

            x_tensor: torch.Tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0).to(next(model.parameters()).device)
            pred_tensor: torch.Tensor = model(x_tensor)
            pred: np.ndarray = pred_tensor.squeeze(0).cpu().numpy()

            pred = pred * global_std + global_mean
            true: np.ndarray = sim_data[t + history + horizon - 1]

            ax.clear()
            ax.plot(np.arange(spatial_points), true, label="True", linewidth=2)
            ax.plot(np.arange(spatial_points), pred, label="Predicted", linestyle="--")
            ax.set_ylim(sim_data.min(), sim_data.max())
            ax.set_title(f"Timestep {t + history + horizon - 1}")
            ax.set_xlabel("Tunnel Position")
            ax.set_ylabel("Temperature (°C)")
            ax.legend()
            ax.grid(True)

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = np.array(Image.open(buf))
            frames.append(image)
            buf.close()

    imageio.mimsave(save_path, frames, fps=fps)
    print(f"Video saved to: {save_path}")
