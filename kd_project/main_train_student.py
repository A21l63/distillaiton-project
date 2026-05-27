import torch

from config import DATA_DIR, STUDENT_CHECKPOINT_PATH
from models import StudentModel
from train import TrainingConfig, train_supervised_cifar10


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = TrainingConfig(
        data_dir=DATA_DIR,
        batch_size=256,
        num_workers=16,
        num_epochs=20,
        learning_rate=0.03,
        optimizer_name="adam",
    )

    student_model = StudentModel().to(device)
    train_supervised_cifar10(
        model=student_model,
        model_name="student model",
        device=device,
        config=config,
        checkpoint_path=STUDENT_CHECKPOINT_PATH,
    )


if __name__ == "__main__":
    main()
