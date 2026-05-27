import torch

from config import DATA_DIR, TEACHER_CHECKPOINT_PATH
from models import TeacherModel
from train import TrainingConfig, train_supervised_cifar10


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = TrainingConfig(
        data_dir=DATA_DIR,
        batch_size=256,
        num_workers=16,
        num_epochs=30,
        learning_rate=1e-4,
        optimizer_name="adamw",
        weight_decay=1e-4,
    )

    teacher_model = TeacherModel().to(device)
    train_supervised_cifar10(
        model=teacher_model,
        model_name="teacher model",
        device=device,
        config=config,
        checkpoint_path=TEACHER_CHECKPOINT_PATH,
        print_trainable_parameters=True,
    )


if __name__ == "__main__":
    main()
