import torch

from data import get_cifar10_dataloaders
from models import StudentModel
from train import fit
from utils import save_checkpoint, count_parameters


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = "./cifar-10"
    student_checkpoint_path = "./student_model"
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.01

    train_loader, test_loader = get_cifar10_dataloaders(data_dir, batch_size)
    student_model = StudentModel().to(device)
    print(f'Number of parameters in student model: {count_parameters(student_model)}')
    optimizer = torch.optim.Adam(params=student_model.parameters(), lr=learning_rate)

    fit(model = student_model, train_loader = train_loader, test_loader = test_loader,
        optimizer = optimizer, device = device, num_epochs = num_epochs)

    save_checkpoint(student_model, student_checkpoint_path)

if __name__ == "__main__":
    main()
