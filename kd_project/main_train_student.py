import torch

from data import get_cifar10_dataloaders
from models import StudentModel
from train import fit
from utils import save_checkpoint, count_parameters
from evaluate import compute_accuracy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = "./cifar-10"
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.001

    train_loader, test_loader = get_cifar10_dataloaders(data_dir, batch_size)
    student_model = StudentModel().to(device)
    print(f"Количество параметров: {count_parameters(student_model)}")
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
    #обучение
    criterion = torch.nn.CrossEntropyLoss() #функция потерь
    for epoch in range(num_epochs):
        print(f"start epoch {epoch}")
        for images, labels in train_loader:
            images = images.to(device) #отправляем данные на устройство
            labels = labels.to(device) #отправляем данные на устройство
            predictions = student_model(images)
            loss = criterion(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"end epoch {epoch}")
    #проверка
    print(compute_accuracy(student_model, test_loader, device))

if __name__ == "__main__":
    main()
