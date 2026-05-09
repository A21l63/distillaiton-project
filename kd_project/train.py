import torch
from tqdm import tqdm

from evaluate import compute_accuracy
from losses import hard_label_loss, distillation_loss


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device: str,
    teacher_model=None,
    temperature: float = 4.0,
    alpha: float = 0.5,
):
    """
    Train model for one epoch.

    If teacher_model is None:
        use hard_label_loss.

    If teacher_model is provided:
        use distillation_loss with frozen teacher outputs.
    """
    model.train()

    if teacher_model is not None:
        teacher_model.eval()

    # TODO: iterate over train_loader
    # TODO: move images and labels to device
    # TODO: compute student/model logits
    # TODO: if teacher_model is None, compute hard_label_loss
    # TODO: else compute teacher logits with torch.no_grad()
    # TODO: compute distillation_loss
    # TODO: backward and optimizer step
    # TODO: track average loss
    ...


def fit(
    model,
    train_loader,
    test_loader,
    optimizer,
    device: str,
    num_epochs: int,
    teacher_model=None,
    temperature: float = 4.0,
    alpha: float = 0.5,
):
    """
    Full training loop.

    Returns:
        history dict with train_loss and test_accuracy.
    """
    history = {
        "train_loss": [],
        "test_accuracy": [],
    }

    # TODO: for each epoch:
    # TODO: train_one_epoch
    # TODO: evaluate accuracy on test_loader
    # TODO: append results to history
    # TODO: print progress

    return history
