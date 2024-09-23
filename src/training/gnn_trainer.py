from copy import deepcopy
import os

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from src.utils.helpers import create_folder_recursively, folder_exists, get_device
from src.model.segcrn import SEGCRN


class PatienceTrainer:
    def __init__(self, patience=5) -> None:
        self.patience = patience
        self.counter = 0
        self.enable = False
        self.min_value = np.inf

    def check_validation(self, new_value):
        if not self.enable:
            if self.min_value > new_value:
                self.counter = 0
                self.min_value = new_value
            else:
                self.counter += 1

            # If the counter equals or greater than the patience set the flag as true
            if self.counter >= self.patience:
                self.enable = True

        return self.enable


class GNNTrainer:
    def __init__(self, config, dataset, model_name, save_model, shuffle=True):
        self.dataset = dataset
        self.model_name = model_name
        self.train_percentage = config["train_percentage"]
        self.val_percentage = config["val_percentage"]
        self.shuffle = shuffle

        self.config = config
        self.save_model = save_model
        self.device = get_device()

        # Load dataset and generate the trainer
        self.dataset, self.train_size, self.val_size, self.test_size = self.load_dataset()
        self.train_loader, self.val_loader, self.test_loader = self.load_subsets(
            self.dataset, self.train_size, self.val_size, self.test_size, config["batch_size"]
        )

        # Load the model
        self.model = self.load_model()
        self.best_model = self.model
        self.best_val = None

        # Early stop
        self.early_stop_checker = PatienceTrainer(patience=self.config["early_stop_patience"])

    def set_model(self, model):
        self.model = model

    def load_subsets(self, dataset, train_size, val_size, test_size, batch_size):
        # Get the indexes
        train_idx, temp_idx = train_test_split(list(range(len(dataset))), test_size=val_size + test_size, shuffle=False)
        val_idx, test_idx = train_test_split(temp_idx, test_size=test_size, shuffle=False)
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        test_subset = Subset(dataset, test_idx)

        # Get the data loaders
        train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=self.shuffle)
        val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=self.shuffle)
        test_loader = DataLoader(dataset=test_subset, batch_size=batch_size, shuffle=self.shuffle)

        return train_loader, val_loader, test_loader

    def load_model(self):
        model = SEGCRN(self.config).to(self.device)
        return model

    def load_dataset(self):
        # Load dataset
        train_size = int(len(self.dataset) * self.train_percentage)
        val_size = int(len(self.dataset) * self.val_percentage)
        test_size = int(len(self.dataset) - train_size - val_size)

        return self.dataset, train_size, val_size, test_size

    def train(self, optimizer, criterion):
        # Configure Tensorboard
        if self.save_model:
            self.init_tensorboard()

        # Train and evaluate the model
        for epoch in range(self.config["num_epochs"]):
            epoch_loss, extra_data = self.train_epoch(self.model, epoch, optimizer, criterion)
            epoch_val_loss, val_extra_data = self.evaluate_epoch(
                self.model, self.val_loader, self.val_size, criterion, "Val"
            )
            epoch_test_loss, test_extra_data = self.evaluate_epoch(
                self.model, self.test_loader, self.test_size, criterion, "Test"
            )

            epoch_mean_loss = epoch_loss / (self.train_size / self.config["batch_size"])
            epoch_mean_val_loss = epoch_val_loss / (self.val_size / self.config["batch_size"])
            epoch_mean_test_loss = epoch_test_loss / (self.test_size / self.config["batch_size"])

            # Check the metric to decide if the model has to stop training
            if self.config["enable_early_stop"]:
                self.early_stop_checker.check_validation(epoch_mean_val_loss)

            print("Epoch: {}/{}.............".format(epoch, self.config["num_epochs"]), end=" ")
            print("Loss: {:.4f} -".format(epoch_mean_loss), end=" ")
            if extra_data is None:
                print("Validation loss: {:.4f} -".format(epoch_mean_val_loss), end=" ")
                print("Test loss: {:.4f}".format(epoch_mean_test_loss))
            else:
                print("Validation loss: {:.4f}".format(epoch_mean_val_loss), end=" ")
                print("Test loss: {:.4f}".format(epoch_mean_test_loss), end=" ")
                for key in extra_data:
                    print("- {}: {:.4f}".format(key.capitalize(), extra_data[key]), end=" ")
                if val_extra_data is not None:
                    for key in val_extra_data:
                        print("- {}: {:.4f}".format(key.capitalize(), val_extra_data[key]), end=" ")
                if test_extra_data is not None:
                    for key in test_extra_data:
                        print("- {}: {:.4f}".format(key.capitalize(), test_extra_data[key]), end=" ")
                print("")

            if self.save_model:
                # Save the model if the validation is the best validation
                if self.best_val is None or epoch_mean_val_loss < self.best_val:
                    print(f"Found a new best model: {epoch_mean_val_loss}")
                    self.best_model = deepcopy(self.model)
                    self.best_val = epoch_mean_val_loss
                    # Save the best model
                    savepoint_path = f"models/traffic/{self.model_name}/training/gnn_best_savepoint.pt"
                    self.save(self.best_model, savepoint_path)

                # Save a savepoint on every iteration
                savepoint_path = f"models/traffic/{self.model_name}/training/savepoints/gnn_savepoint_{epoch}.pt"
                self.save(self.model, savepoint_path)

                # Save the log using tensorboard
                self.log_tensorboard(
                    epoch,
                    epoch_mean_loss,
                    epoch_mean_val_loss,
                    epoch_mean_test_loss,
                    extra_data,
                    val_extra_data,
                    test_extra_data,
                )

            # Check if have to be done an early stop
            if self.early_stop_checker.enable:
                break

    def train_epoch(self, model, epoch, optimizer, criterion) -> float:
        # Save all the training loss to show the epoch loss
        epoch_loss = 0
        epoch_mask_loss = 0
        epoch_mask_layer_loss = [0, 0]
        epoch_mask_use = 0
        epoch_mask_layer_use = [0, 0]
        train_size = len(self.train_loader)
        extra_data = {}

        # Set the model in train mode and start training it
        model.train()
        for i, (target_data, graph_ids, graph_values, times) in enumerate(self.train_loader):
            # Clears existing gradients from previous epoch
            optimizer.zero_grad()
            # Move data to the device
            target_data = target_data[:, -self.config["horizon"] :, :].squeeze(-1)  # noqa: E203
            target_data = target_data.to(self.device)
            graph_ids = graph_ids.to(self.device)
            graph_values = graph_values.to(self.device)
            times = torch.stack(times, dim=1).to(self.device)
            output, masks, mask_losses, _, _ = model(
                graph_ids, graph_values, self.config["enable_explainability"], times
            )

            # Apply inverse transformation
            if hasattr(self.dataset, "scaler"):
                output = self.dataset.scaler.inverse_transform(output)
                target_data = self.dataset.scaler.inverse_transform(target_data)

            # Get the loss by layer
            loss_regression = criterion(output, target_data).mean(dim=(2))
            # Create a deep copy to avoid memory issues
            original_loss = loss_regression.clone()

            # Calculate the loss using the mask
            for mask_loss in mask_losses:
                loss_regression = loss_regression + (original_loss * mask_loss / self.config["cheb_k"])

            # Mean loss regresion
            loss_regression = loss_regression.mean()
            loss_regression.backward()  # Does backpropagation and calculates gradients
            optimizer.step()  # Updates the weights accordingly

            # Calculate the mask loss to show in the log
            mask_layer_losses = []
            for j in range(len(mask_losses)):
                mean_mask_loss = mask_losses[j].mean()
                mask_layer_losses.append(mean_mask_loss)

            # Update the epoch loss
            epoch_loss += original_loss.mean().item()
            # Save the mean mask loss of the iteration
            mean_mask_loss = 0
            mean_mask_use = 0
            # Update the epoch mask layer
            for j in range(len(epoch_mask_layer_loss)):
                epoch_mask_layer_loss[j] += mask_losses[j].mean().item()
                epoch_mask_layer_use[j] += masks[j].mean().item()
                # Increase epoch mask loss
                mean_mask_loss += mask_losses[j].mean().item() / len(epoch_mask_layer_loss)
                mean_mask_use += masks[j].mean().item() / len(epoch_mask_layer_loss)
            # Increase the mask loss of the epoch
            epoch_mask_loss += mean_mask_loss
            epoch_mask_use += mean_mask_use

            if i % self.config["log_every"] == 0 and i != 0:
                print(
                    f"Epoch {epoch}: {i}/{train_size}............. Loss: {loss_regression.item()} - "
                    + f"Mask loss: {mean_mask_loss} - Mask 0: {mask_layer_losses[0]} - Mask 1: {mask_layer_losses[1]}"
                )

        extra_data["Mask loss"] = epoch_mask_loss / (self.train_size / self.config["batch_size"])
        extra_data["Mask loss 0"] = epoch_mask_layer_loss[0] / (self.train_size / self.config["batch_size"])
        extra_data["Mask loss 1"] = epoch_mask_layer_loss[1] / (self.train_size / self.config["batch_size"])
        extra_data["Percentage used"] = epoch_mask_use / (self.train_size / self.config["batch_size"])
        extra_data["Percentage used 0"] = epoch_mask_layer_use[0] / (self.train_size / self.config["batch_size"])
        extra_data["Percentage used 1"] = epoch_mask_layer_use[1] / (self.train_size / self.config["batch_size"])

        return epoch_loss, extra_data

    def evaluate_epoch(self, model, val_loader, val_size, criterion, suffix):
        # Set the model in eval model and get the validation loss
        model.eval()
        epoch_val_loss = 0
        # Mask use
        epoch_mask_val_use = 0
        epoch_mask_layer_val_use = [0, 0]
        extra_data = {}
        with torch.no_grad():
            for i, (target_data, graph_ids, graph_values, times) in enumerate(val_loader):
                # Move data to the device
                target_data = target_data[:, -self.config["horizon"] :, :].clone()  # noqa: E203
                target_data = target_data.to(self.device)
                graph_ids = graph_ids.to(self.device)
                graph_values = graph_values.to(self.device)
                times = torch.stack(times, dim=1).to(self.device)
                output, masks, _, _, _ = model(graph_ids, graph_values, self.config["enable_explainability"], times)

                # Apply inverse transformation
                if hasattr(self.dataset, "scaler"):
                    output = self.dataset.scaler.inverse_transform(output)
                    target_data = self.dataset.scaler.inverse_transform(target_data)

                loss = criterion(output.view(-1), target_data.view(-1)).mean()
                epoch_val_loss += loss.item()

                # Save the mean mask loss of the iteration
                mean_mask_val_use = 0
                # Update the epoch mask layer
                for j in range(len(masks)):
                    # Calculate mean use
                    epoch_mask_layer_val_use[j] += masks[j].mean().item()
                    # Increase epoch mask loss
                    mean_mask_val_use += masks[j].mean().item() / len(epoch_mask_layer_val_use)
                # Increase the mask loss of the epoch
                epoch_mask_val_use += mean_mask_val_use

        extra_data[f"Percentage used - {suffix}"] = epoch_mask_val_use / (val_size / self.config["batch_size"])
        extra_data[f"Percentage used 0 - {suffix}"] = epoch_mask_layer_val_use[0] / (
            val_size / self.config["batch_size"]
        )
        extra_data[f"Percentage used 1 - {suffix}"] = epoch_mask_layer_val_use[1] / (
            val_size / self.config["batch_size"]
        )
        return epoch_val_loss, extra_data

    def get_preditions(self, model, val_loader, val_size):
        # Set the model in eval model and get the validation loss
        model.eval()
        # Save the outputs of the predictions
        all_outputs = []
        # Mask use
        epoch_mask_val_use = 0

        with torch.no_grad():
            for i, (target_data, graph_ids, graph_values, times) in enumerate(val_loader):
                # Move the data to the device
                target_data = target_data[:, -self.config["horizon"] :, :].clone()  # noqa: E203
                target_data = target_data.to(self.device)
                graph_ids = graph_ids.to(self.device)
                graph_values = graph_values.to(self.device)
                times = torch.stack(times, dim=1).to(self.device)
                output, masks, _, _, _ = model(graph_ids, graph_values, self.config["enable_explainability"], times)

                # Apply inverse transformation
                if hasattr(self.dataset, "scaler"):
                    output = self.dataset.scaler.inverse_transform(output)
                    target_data = self.dataset.scaler.inverse_transform(target_data)

                # Save the outputs
                all_outputs.append(output)

                # Save the mean mask loss of the iteration
                mean_mask_val_use = 0
                # Update the epoch mask layer
                for mask in masks:
                    # Increase epoch mask loss
                    mean_mask_val_use += mask.mean().item() / len(masks)
                # Increase the mask loss of the epoch
                epoch_mask_val_use += mean_mask_val_use

        # Convert the list into a tensor
        mask_use = epoch_mask_val_use / (val_size / self.config["batch_size"])
        all_outputs = torch.cat(all_outputs)
        return all_outputs, mask_use

    def init_tensorboard(self):
        tb_path = f"models/traffic/{self.model_name}/training/tensorboard/"
        self.tb_writer = SummaryWriter(log_dir=tb_path)

    def log_tensorboard(self, epoch, loss, val_loss, test_loss, extra_data, val_extra_data, test_extra_data):
        # Register logs with Tensorboard
        self.tb_writer.add_scalar("Loss", loss, global_step=epoch)
        self.tb_writer.add_scalar("Validation loss", val_loss, global_step=epoch)
        self.tb_writer.add_scalar("Test loss", test_loss, global_step=epoch)
        if extra_data is not None:
            for key in extra_data:
                self.tb_writer.add_scalar(f"{key.capitalize()}", extra_data[key], global_step=epoch)
        if val_extra_data is not None:
            for key in val_extra_data:
                self.tb_writer.add_scalar(f"{key.capitalize()}", val_extra_data[key], global_step=epoch)
        if test_extra_data is not None:
            for key in test_extra_data:
                self.tb_writer.add_scalar(f"{key.capitalize()}", test_extra_data[key], global_step=epoch)

    def save(self, model, path):
        folder_path = os.path.dirname(path)
        # Check if the folder exists
        exist = folder_exists(folder_path)
        if not exist:
            create_folder_recursively(folder_path)

        # Save the model
        torch.save(model.state_dict(), path)
