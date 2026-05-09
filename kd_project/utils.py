import matplotlib.pyplot as plt
import torch


def count_parameters(model) -> int:
    """
    Count trainable parameters.
    """
    # TODO: return number of trainable parameters
    ...


def save_checkpoint(model, path: str):
    """
    Save model state_dict.
    """
    # TODO: save state_dict
    ...


def load_checkpoint(model, path: str, device: str):
    """
    Load model state_dict.
    """
    # TODO: load state_dict into model
    ...


def plot_history(history: dict, output_path: str):
    """
    Plot training loss and test accuracy.
    """
    # TODO: plot train_loss and test_accuracy
    # TODO: save figure
    ...
