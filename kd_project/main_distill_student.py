import torch

from data import get_cifar10_dataloaders
from models import StudentModel, TeacherModel
from train import fit
from utils import load_checkpoint, save_checkpoint, count_parameters


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = ...
    teacher_checkpoint_path = ...
    batch_size = ...
    num_epochs = ...
    learning_rate = ...
    temperature = ...
    alpha = ...

    # TODO: create dataloaders
    # TODO: initialize TeacherModel
    # TODO: load teacher checkpoint
    # TODO: freeze teacher parameters
    # TODO: initialize a new StudentModel
    # TODO: create optimizer for student only
    # TODO: train student using fit with teacher_model
    # TODO: save distilled student checkpoint
    ...


if __name__ == "__main__":
    main()
