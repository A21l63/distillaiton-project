import time

import torch
from sklearn.metrics import confusion_matrix

from tqdm.auto import tqdm


def compute_accuracy(
    model,
    dataloader,
    device: str,
    desc: str = "Evaluating",
) -> float:
    """
    Compute classification accuracy.
    """
    was_training = model.training

    model.eval()

    cnt_good_predictions = 0
    cnt_predictions = 0

    eval_pbar = tqdm(
        dataloader,
        desc=desc,
        leave=False,
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for images, labels in eval_pbar:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            predicted_labels = torch.argmax(logits, dim=1)

            cnt_good_predictions += (predicted_labels == labels).sum().item()
            cnt_predictions += labels.size(0)

            running_accuracy = cnt_good_predictions / cnt_predictions

            eval_pbar.set_postfix(
                {
                    "accuracy": f"{running_accuracy:.4f}",
                    "correct": cnt_good_predictions,
                    "total": cnt_predictions,
                }
            )

    if was_training:
        model.train()

    return cnt_good_predictions / cnt_predictions

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
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
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
        for images, _labels in dataloader:
            if i >= num_batches:
                break
            images = images.to(device)
            model(images)
            i += 1
        end_time = time.time()
        return (end_time - start_time) / num_batches
