from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_transforms(train: bool = True):
    """
    Return transforms for CIFAR-10.

    For train:
        basic augmentation may be used.
    For test:
        deterministic transforms.

    Keep this simple.
    """
    """  
    Аугментация разрушает ложные зависимости, Борьба с переобучением (Overfitting)
    CIFAR-10 — это маленькие картинки (всего 32x32 пикселя)
    стоит быть осторожными с параметрами, иначе исказим до каши пикселей
    """

    # Параметры нормализации, стандартные для CIFAR-10. Мы вычитаем «среднее» и делим на «отклонение»
    mean = (0.4914, 0.4822, 0.4465)  # Среднее значение для трех каналов (RGB)
    std = (0.2023, 0.1994, 0.2010)  # Стандартное отклонение для трех каналов (RGB)

    if train:
        # Случайные изменения(аугментация), которые будем двигать + База
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Отражение относительно оси ординат
            transforms.RandomCrop(32, padding=4),
            # Сдвиг картинки (расширяем пустотой на 4, далее случайно выбираем кусок 32*32)
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        # Базовый набор (детерминированный подход)
        transform = transforms.Compose([
            transforms.ToTensor(),  # Конвертация в тензор(превращает картинку из пикселей в матрицу чисел)
            transforms.Normalize(mean, std)  # Нормализация значений (выравнивает числа)
        ])
    return transform


def get_cifar10_dataloaders(
        data_dir: str,
        batch_size: int = 128,  # Количество картинок в пачке
        num_workers: int = 2,  # Количество потоков распараллеливания
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for CIFAR-10.

    Returns:
        train_loader, test_loader
    """

    #  Получаем трансформы (конвейеры обработки)
    train_transform = get_cifar10_transforms(train=True)
    test_transform = get_cifar10_transforms(train=False)

    #  Получаем датасеты
    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                     download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                    download=True, transform=test_transform)

    #  DataLoader — это менеджер очереди, группирует массив(батч) по "batch_size" картинок
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return (train_loader, test_loader)
