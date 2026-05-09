import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def compute_accuracy(model, dataloader, device: str) -> float:
    """
    Compute classification accuracy.
    """
    # TODO: set model.eval()
    # TODO: iterate over dataloader with torch.no_grad()
    # TODO: compute predictions
    # TODO: return accuracy
    ...


def compute_confusion_matrix(model, dataloader, device: str):
    """
    Compute confusion matrix for CIFAR-10.
    """
    # TODO: collect labels and predictions
    # TODO: use sklearn.metrics.confusion_matrix
    ...


def measure_inference_time(
    model,
    dataloader,
    device: str,
    num_batches: int = 20,
) -> float:
    """
    Measure average inference time per batch.

    Keep this simple.
    """
    # TODO: set model.eval()
    # TODO: run several batches with torch.no_grad()
    # TODO: measure elapsed time
    # TODO: return average time per batch
    ...
