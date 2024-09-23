import torch
import numpy as np
from torch.utils.data import Dataset


class TrafficDataset(Dataset):
    def __init__(self, config, dataset_path="data/traffic/PEMSD4/pems04.npz") -> None:
        super().__init__()
        self.config = config
        self.window = self.config["window"]
        self.horizon = self.config["horizon"]

        if "PEMS_BAY" not in dataset_path:
            self.dataset = np.load(dataset_path)["data"][:, :, 0]
        else:
            self.dataset = np.load(dataset_path)["arr_0"]

        # Load and apply scaler
        mean = self.dataset.mean()
        std = self.dataset.std()
        self.scaler = StandardScaler(mean, std)
        self.dataset = self.scaler.transform(self.dataset)

        # Get the graphs ids, that will be always the same
        self.graph_ids = torch.IntTensor(list(range(len(self.dataset[0]))))

        # Convert to float64
        self.graph_values = torch.from_numpy(self.dataset).to(torch.float32)
        self.graph_values = torch.unsqueeze(self.graph_values, 2)

    def __getitem__(self, index):
        indixes = list(range(index, index + 12))
        hours = [int((x % (12 * 24)) / 12) for x in indixes]

        return (
            self.graph_values[index + self.window : index + self.window + self.horizon],  # noqa: E203
            self.graph_ids,
            self.graph_values[index : index + self.window],  # noqa: E203
            hours,
        )

    def __len__(self):
        # Substract the size of the horizon because the last entry of the dataset needs
        # to get the next X (horizon) timesteps
        return len(self.graph_values) - self.config["horizon"] - self.config["window"]


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor) and isinstance(self.mean, np.ndarray):
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean
