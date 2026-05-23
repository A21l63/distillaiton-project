import torch

from data import get_cifar10_dataloaders
from models import StudentModel, TeacherModel
from train import fit
from utils import load_checkpoint, save_checkpoint, count_parameters

teacher_checkpoint_path = "./teacher_model"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = "./cifar-10"
    distill_student_checkpoint_path = "./distill_student_model"
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.01
    temperature = 4.0
    alpha = 0.7

    train_loader, test_loader = get_cifar10_dataloaders(data_dir, batch_size)
    teacher_model = TeacherModel().to(device)
    load_checkpoint(teacher_model, teacher_checkpoint_path, device)
    teacher_model.eval()

    #Заморозка параметров teacher_model
    teacher_model.requires_grad_(False)

    student_model = StudentModel().to(device)
    print(f'Number of parameters in student model: {count_parameters(student_model)}')
    print(f'Number of parameters in teacher model: {count_parameters(teacher_model)}')
    optimizer = torch.optim.Adam(params=student_model.parameters(), lr=learning_rate)
    fit(model=student_model, train_loader=train_loader, test_loader=test_loader,optimizer=optimizer,
        device=device, num_epochs=num_epochs, teacher_model = teacher_model, temperature = temperature, alpha = alpha)

    save_checkpoint(student_model, distill_student_checkpoint_path)

if __name__ == "__main__":
    main()
