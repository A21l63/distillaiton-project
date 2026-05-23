import torch
import torch.nn.functional as F

def hard_label_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Standard cross-entropy loss.

    logits: [B, num_classes]
    labels: [B]
    """
    return F.cross_entropy(input = logits, target = labels)

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
    hard_loss = hard_label_loss(logits = student_logits, labels = labels)

    teacher_probs = F.softmax(teacher_logits / temperature, dim = 1)

    student_log_probs = F.log_softmax(student_logits / temperature, dim = 1)

    # Следующая строка эквивалентна: soft_loss = torch.sum(teacher_probs * (teacher_probs.log() - student_log_probs)) / teacher_probs.size()[0]
    soft_loss = torch.nn.KLDivLoss(reduction='batchmean')(input = student_log_probs, target = teacher_probs)

    loss = (alpha * hard_loss) + (1 - alpha) * soft_loss * (temperature ** 2)

    return loss
