import torch
from pathlib import Path


def get_device(show_info=True) -> torch.device:
    # If there is a GPU available, then use the GPU.
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
        if show_info:
            print("GPU is available")
    else:
        device = torch.device("cpu")
        if show_info:
            print("GPU not available, CPU used")
    return device


def folder_exists(folder_path):
    my_folder = Path(folder_path)
    if my_folder.is_dir():
        return True
    else:
        return False


def create_folder_recursively(folder_path):
    output_path = Path(folder_path)
    output_path.mkdir(exist_ok=True, parents=True)
