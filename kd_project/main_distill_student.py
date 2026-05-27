import torch

from config import DATA_DIR, DISTILLED_STUDENT_CHECKPOINT_PATH, TEACHER_CHECKPOINT_PATH
from models import StudentModel, TeacherModel
from train import TrainingConfig, fit_cifar10
from utils import load_checkpoint, save_checkpoint, count_parameters


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = TrainingConfig(
        data_dir=DATA_DIR,
        batch_size=128,
        num_workers=16,
        num_epochs=30,
        learning_rate=0.03,
        optimizer_name="adam",
    )
    temperature = 3.0
    alpha = 0.8

    teacher_model = TeacherModel().to(device)
    load_checkpoint(teacher_model, TEACHER_CHECKPOINT_PATH, device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)

    student_model = StudentModel().to(device)
    print(f"Number of parameters in student model: {count_parameters(student_model)}")
    print(f"Number of parameters in teacher model: {count_parameters(teacher_model)}")

    fit_cifar10(
        model=student_model,
        device=device,
        config=config,
        teacher_model=teacher_model,
        temperature=temperature,
        alpha=alpha,
    )

    save_checkpoint(student_model, DISTILLED_STUDENT_CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
