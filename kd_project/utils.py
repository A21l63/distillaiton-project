import matplotlib.pyplot as plt
import torch


def count_parameters(model) -> int:
    """
    Count trainable parameters.
    """
    cnt = 0
    for p in model.parameters():
        if p.requires_grad: #проверка на то, что для этого параметра нужно вычислять градиенты, то есть он нам интересен, изучаем его
            cnt += p.numel() #p - вектор, p.numel() - размерность/количество элементов
    return cnt


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
