import torch
from tqdm import tqdm

from evaluate import compute_accuracy
from losses import hard_label_loss, distillation_loss

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device: str,
    teacher_model=None,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> float:
    """
    Train model for one epoch.

    If teacher_model is None:
        use hard_label_loss.

    If teacher_model is provided:
        use distillation_loss with frozen teacher outputs.
    """
    model.train()

    if teacher_model is not None:
        teacher_model.eval()

    lm_count = 0
    loss_sum = 0
    for images, labels in train_loader:
        images = images.to(device)  # отправляем данные на устройство
        labels = labels.to(device)  # отправляем данные на устройство
        logits = model(images)

        if teacher_model is None:
            loss = hard_label_loss(logits, labels)
        else:
            with torch.no_grad():
                teacher_logits = teacher_model(images)
            loss = distillation_loss(logits, teacher_logits, labels, temperature, alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_sum += loss.item()

    return loss_sum / lm_count

def fit(
    model,
    train_loader,
    test_loader,
    optimizer,
    device: str,
    num_epochs: int,
    teacher_model=None,
    temperature: float = 4.0,
    alpha: float = 0.5,
):
    """
    Full training loop.

    Returns:
        history dict with train_loss and test_accuracy.
    """
    history = {
        "train_loss": [],
        "test_accuracy": [],
    }

    tracking_tqdm = tqdm(range(num_epochs), leave = True) #Используется для вывода информации по эпохам.
    for _e in tracking_tqdm:
        loss_mean = train_one_epoch(model, train_loader, optimizer, device, teacher_model, temperature, alpha)
        history["test_accuracy"].append(compute_accuracy(model, test_loader, device))
        history["train_loss"].append(loss_mean)
        tracking_tqdm.set_description(f"Epoch {tracking_tqdm.n}, loss_mean={history["train_loss"]}, test_accuracy={history['test_accuracy']}")

    return history
