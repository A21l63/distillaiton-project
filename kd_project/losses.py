import torch
import torch.nn.functional as F


def hard_label_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Standard cross-entropy loss.

    logits: [B, num_classes]
    labels: [B]
    """
    # TODO: compute cross entropy loss
    ...


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Knowledge distillation loss.

    Combine:
        hard loss: CE(student_logits, labels)
        soft loss: KL divergence between softened student and teacher distributions

    Use temperature for both teacher and student logits.

    Recommended formula:
        loss = alpha * hard_loss + (1 - alpha) * soft_loss * temperature^2

    Important:
        - teacher probabilities should use softmax(teacher_logits / T)
        - student log probabilities should use log_softmax(student_logits / T)
        - use KL divergence
    """
    # TODO: compute hard loss
    hard_loss = ...

    # TODO: compute softened teacher probabilities
    teacher_probs = ...

    # TODO: compute softened student log-probabilities
    student_log_probs = ...

    # TODO: compute KL divergence soft loss
    soft_loss = ...

    # TODO: combine losses
    loss = ...

    return loss
