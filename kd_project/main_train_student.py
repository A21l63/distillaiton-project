import torch

from data import get_cifar10_dataloaders
from models import StudentModel
from train import fit
from utils import save_checkpoint, count_parameters


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = ...
    batch_size = ...
    num_epochs = ...
    learning_rate = ...

    # TODO: create dataloaders
    # TODO: initialize StudentModel
    # TODO: print number of parameters
    # TODO: create optimizer
    # TODO: train using fit without teacher_model
    # TODO: save checkpoint
    ...


if __name__ == "__main__":
    main()
