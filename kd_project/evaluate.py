import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def compute_accuracy(model, dataloader, device: str) -> float:
    """
    Compute classification accuracy.
    """
    with torch.no_grad(): #отключение вычисления градиентов
        model.eval() #говорим модели ничего не запоминать
        cnt_good_predictions = 0
        cnt_predictions = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
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
        model.train() #возвращаем модель в режим обучения
        return cnt_good_predictions / cnt_predictions #возвращаем долю правильных ответов


def compute_confusion_matrix(model, dataloader, device: str):
    """
    Compute confusion matrix for CIFAR-10.
    """
    with torch.no_grad(): #отключение вычисления градиентов
        model.eval() #говорим модели ничего не запоминать
        all_labels = []
        all_predictions = []
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            all_labels.extend(labels.cpu().numpy())
            predictions = model(images)
            ans = []
            for s in predictions:
                idx = 0
                for i in range(len(s)):
                    if s[i] > s[idx]:
                        idx = i
                ans.append(idx)
            all_predictions.extend(ans)
        cm = confusion_matrix(all_labels, all_predictions)
        return cm


def measure_inference_time(
    model,
    dataloader,
    device: str,
    num_batches: int = 20,
) -> float:
    """
    Measure average inference time per batch.

    Keep this simple.
    """
    with torch.no_grad(): #отключение вычисления градиентов
        model.eval() #говорим модели ничего не запоминать
        start_time = time.time()
        i = 0
        for images, labels in dataloader:
            if i >= num_batches:
                break
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            i += 1
        end_time = time.time()
        return (end_time - start_time) / num_batches
