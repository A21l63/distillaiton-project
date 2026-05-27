import matplotlib.pyplot as plt
import torch


def count_parameters(model) -> int:
    """
    Count trainable parameters.
    """
    cnt = 0
    for p in model.parameters():
        if p.requires_grad:  # проверка на то, что для этого параметра нужно вычислять градиенты, то есть он нам интересен, изучаем его
            cnt += p.numel()  # p - вектор, p.numel() - размерность/количество элементов
    return cnt


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
