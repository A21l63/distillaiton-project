from dataclasses import dataclass
from typing import Iterable

import torch
from tqdm.auto import tqdm

from data import get_cifar10_dataloaders
from evaluate import compute_accuracy
from losses import hard_label_loss, distillation_loss
from utils import (
    build_cosine_warmup_scheduler,
    count_parameters,
    count_trainable_parameters,
    save_checkpoint,
)


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: str = "./cifar-10"
    batch_size: int = 128
    num_workers: int = 2
    num_epochs: int = 20
    learning_rate: float = 0.03
    optimizer_name: str = "adam"
    weight_decay: float = 0.0
    scheduler_step: str = "batch"
    max_grad_norm: float | None = 1.0
    use_amp: bool = False


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float = 0.0,
):
    trainable_parameters = [p for p in parameters if p.requires_grad]

    if optimizer_name == "adam":
        return torch.optim.Adam(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            trainable_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def fit_cifar10(
    model,
    device: str,
    config: TrainingConfig,
    teacher_model=None,
    temperature: float = 4.0,
    alpha: float = 0.5,
):
    train_loader, test_loader = get_cifar10_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    optimizer = build_optimizer(
        parameters=model.parameters(),
        optimizer_name=config.optimizer_name,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    return fit(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=config.num_epochs,
        teacher_model=teacher_model,
        temperature=temperature,
        alpha=alpha,
        scheduler_step=config.scheduler_step,
        max_grad_norm=config.max_grad_norm,
        use_amp=config.use_amp,
    )


def train_supervised_cifar10(
    model,
    model_name: str,
    device: str,
    config: TrainingConfig,
    checkpoint_path: str,
    print_trainable_parameters: bool = False,
):
    print(f"Number of parameters in {model_name}: {count_parameters(model)}")

    if print_trainable_parameters:
        trainable_count = count_trainable_parameters(model)
        print(f"Number of trainable parameters in {model_name}: {trainable_count}")

    history = fit_cifar10(model=model, device=device, config=config)
    save_checkpoint(model, checkpoint_path)

    return history


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device: str,
    teacher_model=None,
    temperature: float = 4.0,
    alpha: float = 0.5,
    epoch: int | None = None,
    num_epochs: int | None = None,
    scheduler=None,
    scheduler_step: str = "batch",  # "epoch" or "batch"
    max_grad_norm: float | None = 1.0,
    use_amp: bool = False,
) -> float:
    """
    Train model for one epoch.

    If teacher_model is None:
        use hard_label_loss.

    If teacher_model is provided:
        use distillation_loss with frozen teacher outputs.

    Args:
        scheduler:
            Optional PyTorch LR scheduler.

        scheduler_step:
            "batch" if scheduler should step after every optimizer step.
            "epoch" if scheduler should step after each epoch inside fit.

        max_grad_norm:
            If not None, clips gradients using torch.nn.utils.clip_grad_norm_.

        use_amp:
            Enables automatic mixed precision on CUDA.
    """
    model.train()

    if teacher_model is not None:
        teacher_model.eval()

    if scheduler_step not in {"epoch", "batch"}:
        raise ValueError("scheduler_step must be either 'epoch' or 'batch'")

    amp_enabled = use_amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    loss_sum = 0.0
    batch_count = 0

    if epoch is not None and num_epochs is not None:
        epoch_prefix = f"Epoch {epoch + 1}/{num_epochs}"
    elif epoch is not None:
        epoch_prefix = f"Epoch {epoch + 1}"
    else:
        epoch_prefix = "Training"

    batch_pbar = tqdm(
        train_loader,
        desc=epoch_prefix,
        leave=False,
        dynamic_ncols=True,
    )

    for images, labels in batch_pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images)

            if teacher_model is None:
                loss = hard_label_loss(logits, labels)
            else:
                with torch.no_grad():
                    teacher_logits = teacher_model(images)

                loss = distillation_loss(
                    logits,
                    teacher_logits,
                    labels,
                    temperature,
                    alpha,
                )

        scaler.scale(loss).backward()

        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )
        else:
            grad_norm = None

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None and scheduler_step == "batch":
            scheduler.step()

        batch_count += 1
        loss_sum += loss.item()
        running_loss = loss_sum / batch_count

        current_lr = optimizer.param_groups[0]["lr"]

        postfix = {
            "batch_loss": f"{loss.item():.4f}",
            "mean_loss": f"{running_loss:.4f}",
            "lr": f"{current_lr:.2e}",
        }

        if grad_norm is not None:
            postfix["grad_norm"] = f"{float(grad_norm):.2f}"

        batch_pbar.set_postfix(postfix)

    return loss_sum / batch_count


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
    scheduler=None,
    scheduler_step: str = "batch",  # "epoch" or "batch"
    max_grad_norm: float | None = 1.0,
    use_amp: bool = False,
):
    """
    Full training loop.

    Returns:
        history dict with train_loss, test_accuracy, and lr.
    """
    if scheduler_step not in {"epoch", "batch"}:
        raise ValueError("scheduler_step must be either 'epoch' or 'batch'")

    history = {
        "train_loss": [],
        "test_accuracy": [],
        "lr": [],
    }

    print(f"model trains on {device}")

    epoch_pbar = tqdm(
        range(num_epochs),
        desc="Epochs",
        leave=True,
        dynamic_ncols=True,
    )
    if scheduler is None:
        scheduler = build_cosine_warmup_scheduler(
            optimizer=optimizer,
            num_epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            warmup_epochs=1.0,
            min_lr_ratio=0.05,
        )

    for epoch in epoch_pbar:
        loss_mean = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            teacher_model=teacher_model,
            temperature=temperature,
            alpha=alpha,
            epoch=epoch,
            num_epochs=num_epochs,
            scheduler=scheduler,
            scheduler_step=scheduler_step,
            max_grad_norm=max_grad_norm,
            use_amp=use_amp,
        )

        if scheduler is not None and scheduler_step == "epoch":
            scheduler.step()

        test_accuracy = compute_accuracy(model, test_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(loss_mean)
        history["test_accuracy"].append(test_accuracy)
        history["lr"].append(current_lr)

        epoch_pbar.set_postfix(
            {
                "train_loss": f"{loss_mean:.4f}",
                "test_acc": f"{test_accuracy:.4f}",
                "lr": f"{current_lr:.2e}",
            }
        )

        epoch_pbar.write(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={loss_mean:.4f} | "
            f"test_accuracy={test_accuracy:.4f} | "
            f"lr={current_lr:.2e}"
        )

    return history
