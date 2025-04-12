import random
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from torch.utils.data import Dataset, DataLoader, random_split
from torch.jit import RecursiveScriptModule
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
# T0099.csv

class TunnelTemperatureDataset(Dataset):
    """
    Custom Dataset for loading tunnel temperature simulations.
    Each CSV is a simulation with shape [time, space] => [1000, 1500].
    """

    def __init__(self, csv_folder: str, history: int = 30, horizon: int = 10, spatial_stride: int = 1) -> None:
        """
        Args:
            csv_folder (str): Path to main data folder containing subfolders with "T0099.csv".
            history (int): Number of past seconds to use as input.
            horizon (int): Number of seconds into the future to predict.
            spatial_stride (int): Downsampling factor for spatial resolution.
        """
        self.csv_folder = csv_folder
        self.history = history
        self.horizon = horizon
        self.spatial_stride = spatial_stride

        self.simulations = []
        self.samples = []

        print(f"Looking for CSV files in: {csv_folder}")

        # Traverse subdirectories to find all "T0099.csv"
        for root, _, files in os.walk(csv_folder):
            for file in files:
                if file == "T0099.csv":
                    full_path = os.path.join(root, file)
                    print(f"Found: {full_path}")

                    try:
                        data = np.loadtxt(full_path, delimiter=",", dtype=np.float32)
                    except Exception as e:
                        print(f"Error reading {full_path}: {e}")
                        continue

                    # Ensure shape and remove first column if needed
                    if data.shape[1] > 1501:
                        data = data[:, :1501]
                    data = data[:, 1:]  # remove first column (e.g., time)

                    if data.shape[1] == 1500:
                        downsampled_data = data[:, ::spatial_stride]
                        self.simulations.append(downsampled_data)
                        print(f"Loaded simulation with shape: {downsampled_data.shape}")
                    else:
                        print(f"Skipping {full_path}: unexpected number of columns ({data.shape[1]})")

        if not self.simulations:
            raise RuntimeError("No valid simulations found with 1500 spatial points.")

        self.total_time_steps = self.simulations[0].shape[0]
        self.num_points = self.simulations[0].shape[1]

        for sim_idx, sim in enumerate(self.simulations):
            for t in range(self.total_time_steps - history - horizon):
                self.samples.append((sim_idx, t))

        print(f"Total simulations loaded: {len(self.simulations)}")
        print(f"Total samples generated: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sim_idx, t = self.samples[idx]
        sim = self.simulations[sim_idx]

        x = sim[t : t + self.history]
        y = sim[t + self.history : t + self.history + self.horizon]

        # Debug print to verify shapes
        if x.shape != (self.history, self.num_points) or y.shape != (self.horizon, self.num_points):
            print(f"Warning: Unexpected shape at idx {idx} (sim {sim_idx}, t={t})")
            print(f"x shape: {x.shape}, expected: ({self.history}, {self.num_points})")
            print(f"y shape: {y.shape}, expected: ({self.horizon}, {self.num_points})")

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def get_dataloaders(
    csvs_path: str,
    batch_size: int,
    history: int,
    horizon: int,
    spatial_stride: int,
    num_workers: int = 4
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset: TunnelTemperatureDataset = TunnelTemperatureDataset(csvs_path, history, horizon, spatial_stride)
    print("dataset", len(dataset))
    train_size: int = int(0.8 * len(dataset))
    val_size: int = int(0.1 * len(dataset))
    test_size: int = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader: DataLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader: DataLoader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    test_loader: DataLoader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    return train_loader, val_loader, test_loader


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
