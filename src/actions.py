import json

import torch
from torch import nn

from src.model.segcrn import SEGCRN
from src.data.traffic_dataset import TrafficDataset
from src.training.gnn_trainer import GNNTrainer
from src.utils.helpers import get_device


def load_config(dataset_name):
    # Load configuration
    with open(f"config/{dataset_name}.json", "r") as file:
        config = json.load(file)
    return config


def load_dataset(config, dataset_name):
    if dataset_name == "PEMSD3":
        dataset = TrafficDataset(config, dataset_path="data/traffic/PEMSD3/pems03.npz")
    elif dataset_name == "PEMSD4":
        dataset = TrafficDataset(config, dataset_path="data/traffic/PEMSD4/pems04.npz")
    elif dataset_name == "PEMSD7":
        dataset = TrafficDataset(config, dataset_path="data/traffic/PEMSD7/pems07.npz")
    elif dataset_name == "PEMSD8":
        dataset = TrafficDataset(config, dataset_path="data/traffic/PEMSD8/pems08.npz")
    elif dataset_name == "PEMS_BAY":
        dataset = TrafficDataset(config, dataset_path="data/traffic/PEMS_BAY/pems-bay.npz")
    else:
        raise Exception("Invalid dataset name")

    return dataset


def start_training(dataset_name, save_model=True):
    # Load config and dataset
    config = load_config(dataset_name)
    dataset = load_dataset(config, dataset_name)

    # Define the trainer
    trainer = GNNTrainer(config, dataset, dataset_name, save_model)

    # The reduction is set to "none" to be able to calculate the mask loss by layer
    criterion = nn.L1Loss(reduction="none")

    # Set the optimizer
    optimizer = torch.optim.RAdam(trainer.model.parameters(), lr=config["learning_rate"])

    # Start training
    trainer.train(optimizer, criterion)


def execute_validation(config, savepoint_path, trainer):
    # Get the device we are using
    device = get_device(show_info=False)

    # The reduction is set to "none" to be able to calculate the mask loss by layer
    criterion = nn.L1Loss(reduction="none")

    del trainer.model
    model = SEGCRN(config).to(device)
    model.load_state_dict(torch.load(savepoint_path))
    trainer.set_model(model)

    # Evaluate
    epoch_test_loss, extra_data = trainer.evaluate_epoch(
        model, trainer.test_loader, trainer.test_size, criterion, suffix="test"
    )
    epoch_mean_test_loss = epoch_test_loss / (trainer.test_size / config["batch_size"])
    sparsity = 1 - extra_data["Percentage used - test"]
    return epoch_mean_test_loss, sparsity


def start_evaluation(dataset_name):
    # Load config and dataset
    config = load_config(dataset_name)
    dataset = load_dataset(config, dataset_name)

    # Define the trainer
    trainer = GNNTrainer(config, dataset, dataset_name, False, shuffle=False)

    # Evaluate no explainable version
    config["enable_explainability"] = False
    savepoint_path = f"models/traffic/{dataset_name}/evaluation/no_explainable/gnn_best_savepoint.pt"
    epoch_mean_test_loss, _ = execute_validation(config, savepoint_path, trainer)
    print(f"No explainable ({dataset_name}): {epoch_mean_test_loss}")

    # Evaluate explainable version
    config["enable_explainability"] = True
    savepoint_path = f"models/traffic/{dataset_name}/evaluation/explainable/gnn_best_savepoint.pt"
    epoch_mean_test_loss, sparsity = execute_validation(config, savepoint_path, trainer)
    print(f"Explainable ({dataset_name} - Sparsity: {sparsity:0.2f}): {epoch_mean_test_loss}")


def get_outputs(config, savepoint_path, trainer, inverse_mask):
    # Get the device we are using
    device = get_device(show_info=False)

    del trainer.model
    model = SEGCRN(config, inverse_mask=inverse_mask).to(device)
    model.load_state_dict(torch.load(savepoint_path))
    trainer.set_model(model)

    # Get the predictions
    outputs, mask_use = trainer.get_preditions(model, trainer.test_loader, trainer.test_size)
    sparsity = 1 - mask_use
    return outputs, sparsity


def start_fidelity(dataset_name):
    # Load config and dataset
    config = load_config(dataset_name)
    dataset = load_dataset(config, dataset_name)

    # Define the trainer
    trainer = GNNTrainer(config, dataset, dataset_name, False, shuffle=False)

    # Get the predictions of the model without explainability
    config["enable_explainability"] = False
    savepoint_path = f"models/traffic/{dataset_name}/evaluation/no_explainable/gnn_best_savepoint.pt"
    original_outputs, _ = get_outputs(config, savepoint_path, trainer, False)

    # Get the predictions with the mask and the inverse mask
    config["enable_explainability"] = True
    savepoint_path = f"models/traffic/{dataset_name}/evaluation/explainable/gnn_best_savepoint.pt"
    infidelity_outputs, _ = get_outputs(config, savepoint_path, trainer, True)
    fidelity_outputs, sparsity = get_outputs(config, savepoint_path, trainer, False)

    # Calculate the fidelity and infidelity
    criterion = nn.L1Loss()
    fidelify = criterion(original_outputs, infidelity_outputs)
    infidelity = criterion(original_outputs, fidelity_outputs)
    print(f"Fidelity ({dataset_name}): {fidelify}")
    print(f"Infidelity ({dataset_name} - Sparsity: {sparsity:0.2f}): {infidelity}")
