import torch

from data import get_cifar10_dataloaders
from models import TeacherModel
from train import fit
from utils import save_checkpoint, count_parameters


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = "./cifar-10"
    teacher_checkpoint_path = "./teacher_model"
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.01

    train_loader, test_loader = get_cifar10_dataloaders(data_dir, batch_size)
    teacher_model = TeacherModel().to(device)
    print(f'Number of parameters in teacher model: {count_parameters(teacher_model)}')
    optimizer = torch.optim.Adam(params=teacher_model.parameters(), lr=learning_rate)

    fit(model = teacher_model, train_loader = train_loader, test_loader = test_loader,
        optimizer = optimizer, device = device, num_epochs = num_epochs)

    save_checkpoint(teacher_model, teacher_checkpoint_path)

if __name__ == "__main__":
    main()
