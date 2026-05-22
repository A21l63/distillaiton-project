import torch

from data import get_cifar10_dataloaders
from models import StudentModel, TeacherModel
from train import fit
from utils import load_checkpoint, save_checkpoint, count_parameters


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = "./cifar-10"
    teacher_checkpoint_path = "./teacher_model"
    distill_student_checkpoint_path = "./distill_student_model"
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.01
    temperature = 4.0
    alpha = 0.5

    train_loader, test_loader = get_cifar10_dataloaders(data_dir, batch_size)
    teacher_model = TeacherModel().to(device)
    load_checkpoint(teacher_checkpoint_path, teacher_model, next(teacher_model.parameters()).device)

    #Заморозка параметров teacher_model
    for param in teacher_model.parameters():
        param.requires_grad = False

    student_model = StudentModel().to(device)
    optimizer = torch.optim.Adam(params=student_model.parameters(), lr=learning_rate)
    fit(model=student_model, train_loader=train_loader, test_loader=test_loader,optimizer=optimizer,
        device=device, num_epochs=num_epochs, teacher_model = teacher_model, temperature = temperature, alpha = alpha)

    save_checkpoint(student_model, teacher_checkpoint_path)


    print(f'Number of parameters in teacher model: {count_parameters(teacher_model)}')
    optimizer = torch.optim.Adam(params=teacher_model.parameters(), lr=learning_rate)

    fit(model=teacher_model, train_loader=train_loader, test_loader=test_loader,
        optimizer=optimizer, device=device, num_epochs=num_epochs)

    save_checkpoint(teacher_model, distill_student_checkpoint_path)

if __name__ == "__main__":
    main()
