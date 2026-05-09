import torch

from data import get_cifar10_dataloaders
from models import StudentModel, TeacherModel
from evaluate import compute_accuracy, measure_inference_time, compute_confusion_matrix
from utils import load_checkpoint, count_parameters


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = ...

    student_checkpoint_path = ...
    teacher_checkpoint_path = ...
    distilled_student_checkpoint_path = ...

    # TODO: create test dataloader
    # TODO: load StudentModel baseline
    # TODO: load TeacherModel
    # TODO: load distilled StudentModel
    # TODO: compute accuracy, parameter count and inference time
    # TODO: print comparison table
    # TODO: compute confusion matrix for selected models
    ...


if __name__ == "__main__":
    main()
