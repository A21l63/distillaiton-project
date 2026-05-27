import torch
from torch import nn


class ClassificationModel(nn.Module):
    """
    Base class for image classification models.

    Child classes must implement forward().
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: [B, C, H, W]

        Output:
            logits: [B, num_classes]
        """
        raise NotImplementedError


class StudentModel(ClassificationModel):
    """
    Small CNN student model for CIFAR-10.

    Expected input:
        x: [B, 3, 32, 32]

    Expected output:
        logits: [B, 10]
    """

    def __init__(self, num_classes: int = 10,
                 in_ch1: int = 3,  # Входные/выходные каналы, изначально RGB(3)
                 out_ch1_in_ch2: int = 32,
                 out_ch2_in_ch3: int = 64,
                 out_ch3: int = 128,
                 kernel_size: int = 3,  # Размер скользящего окна в свертке
                 padding: int = 1,  # Параметр, отвечающий за неизменность размера матрицы после свертки
                 pool_size: int = 2,  # Размер сжатия (каждые 2*2 пикселя в 1)
                 target_output_size: tuple[int, int] = (1, 1)):  # Усредняет все пиксели по каналам
        super().__init__(num_classes=num_classes)

        # example structure:
        # Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
        # Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
        # Conv2d -> BatchNorm2d -> ReLU -> AdaptiveAvgPool2d
        # Linear -> logits

        # Треш контент
        self.features = nn.Sequential(
            # Блок 1: 32x32 -> 16x16
            nn.Conv2d(in_ch1, out_ch1_in_ch2, kernel_size=kernel_size, padding=padding),  # Свертка
            nn.BatchNorm2d(out_ch1_in_ch2),  # Стандартизация значений
            nn.ReLU(),  # Убирает отрицательные значения
            nn.MaxPool2d(pool_size),  # Упрощение (из каждого блока 2*2 берем только макс)

            # Блок 2: 16x16 -> 8x8
            nn.Conv2d(out_ch1_in_ch2, out_ch2_in_ch3, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch2_in_ch3),
            nn.ReLU(),
            nn.MaxPool2d(pool_size),

            # Блок 3: 8x8 -> 1x1 (благодаря AdaptiveAvgPool2d)
            nn.Conv2d(out_ch2_in_ch3, out_ch3, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(target_output_size)  # Финалит тензор в вид с 1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # Убирает все ненужные размерности
            nn.Linear(out_ch3, num_classes)  # Определяет класс картинки?
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)  # Обработка картинки
        logits = self.classifier(x)  # Классификация
        return logits


class TeacherModel(ClassificationModel):
    """
    Teacher model based on pretrained ResNet18.

    Steps:
        1. Load torchvision.models.resnet18 with pretrained ImageNet weights.
        2. Replace final fully-connected layer with Linear(in_features, num_classes).
        3. Return logits for CIFAR-10 classes.
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__(num_classes=num_classes)

        if num_classes != 10:
            raise ValueError(
                "TeacherModel uses a CIFAR-10 checkpoint and supports only 10 classes"
            )

        # List all available model entrypoints in a specific repo
        # models = torch.hub.list('chenyaofo/pytorch-cifar-models')
        # print(models)

        self.backbone = torch.hub.load(
            "chenyaofo/pytorch-cifar-models",
            "cifar10_mobilenetv2_x1_4",
            pretrained=pretrained,
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
