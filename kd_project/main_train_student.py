import torch

from data import get_cifar10_dataloaders
from models import StudentModel
from train import fit
from utils import save_checkpoint, count_parameters


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
    with torch.no_grad(): #отключение вычисления градиентов
        student_model.eval() #говорим модели ничего не запоминать
        cnt_good_predictions = 0
        cnt_predictions = 0
        for images, labels in test_loader:
            predictions = student_model(images)
            ans = [] #массив номеров самых вероятных классов для каждого батча
            for s in predictions:
                idx = 0
                for i in range(len(s)):
                    if s[i] > s[idx]:
                        idx = i
                ans.append(idx)
            for i in range(len(ans)):
                if ans[i] == labels[i]:
                    cnt_good_predictions += 1
            cnt_predictions += len(ans)
        print(cnt_good_predictions / cnt_predictions) #выводим долю правильных ответов
        student_model.train() #возвращаем модель в режим обучения

if __name__ == "__main__":
    main()
