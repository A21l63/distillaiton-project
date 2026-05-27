import math

import matplotlib.pyplot as plt
import torch


def count_parameters(model) -> int:
    """
    Count all parameters.
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, path: str):
    """
    Save model state_dict.
    """
    # state_dict содержит веса слоёв, bias, обучаемые параметры модели
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path: str, device: str):
    """
    Load model state_dict.
    """
    # device можно задать модели самостоятельно model.to(device),
    # либо воспользоваться next(model.parameters()).device, чтобы узнать его, скорее всего у вас - "cpu"
    weights = torch.load(path, map_location=device)
    model.load_state_dict(weights)
    return model


def build_cosine_warmup_scheduler(
    optimizer,
    num_epochs: int,
    steps_per_epoch: int,
    warmup_epochs: float = 1.0,
    min_lr_ratio: float = 0.0,
):
    """
    Linear warmup followed by cosine decay.

    Args:
        optimizer:
            PyTorch optimizer.

        num_epochs:
            Total number of training epochs.

        steps_per_epoch:
            Number of optimizer steps per epoch.

        warmup_epochs:
            Number of warmup epochs. Can be fractional, e.g. 0.5.

        min_lr_ratio:
            Final LR as a fraction of the initial LR.
            0.0 means decay close to zero.
            0.1 means decay to 10% of initial LR.
    """
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    if warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be smaller than total_steps")

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def plot_history(history: dict, output_path: str,
                 figsize: tuple[int, int] = (12, 5),
                 color_loss: str = 'yellow',
                 color_accur: str = 'green'
                 ):
    """
    Plot training loss and test accuracy.
    """
    # Зафиксируем две таблицы, так как наша функция будет работать только с training loss и test accuracy
    rows: int = 2
    cols: int = 1
    # Создаем общее окно нужного размера
    fig, (ax1, ax2) = plt.subplots(rows, cols, figsize=figsize)

    # Верхний график - train_loss
    ax1.plot(history['train_loss'], label='Train Loss', color=color_loss)
    ax1.set_title('Train Loss')
    ax1.set_xlabel('Epoch')  # Эпохи - ось X
    ax1.set_ylabel('Loss')  # Значения потерь - ось Y
    ax1.grid(True)  # Включаем сетку
    ax1.legend()  # Показываем легенду (подписи линий)

    # Нижний график - test accuracy
    ax2.plot(history['test_accuracy'], label='Test accuracy', color=color_accur)
    ax2.set_title('Test accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')  # Точность - ось Y
    ax2.grid(True)
    ax2.legend()

    # Автоматически выравнивает отступы, чтобы подписи не накладывались
    plt.tight_layout()

    fig.savefig(output_path)
    plt.close()
