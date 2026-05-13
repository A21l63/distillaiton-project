import torch
from torch import nn
from torchvision import models


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

    def __init__(self, num_classes: int = 10):
        super().__init__(num_classes=num_classes)

        # example structure:
        # Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
        # Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
        # Conv2d -> BatchNorm2d -> ReLU -> AdaptiveAvgPool2d
        # Linear -> logits

        # Треш контент
        self.features = nn.Sequential(
            # Блок 1: 32x32 -> 16x16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Свертка (in_channel, out_channel, геометрические параметры)
            nn.BatchNorm2d(32),  # Стандартизация значений
            nn.ReLU(),  # Убирает отрицательные значения
            nn.MaxPool2d(2),  # Упрощение (из каждого блока 2*2 берем только макс)

            # Блок 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Блок 3: 8x8 -> 1x1 (благодаря AdaptiveAvgPool2d)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Финалит тензор в вид с 1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # Убирает все ненужные размерности
            nn.Linear(128, num_classes)  # Определяет класс картинки?
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

        # TODO: load pretrained ResNet18
        # TODO: replace final fc layer
        self.backbone = ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: call self.backbone(x)
        ...
