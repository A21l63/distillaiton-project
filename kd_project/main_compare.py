import torch
import pandas as pd

from config import DATA_DIR, TEACHER_CHECKPOINT_PATH
from data import get_cifar10_dataloaders
from models import TeacherModel
from evaluate import compute_accuracy, measure_inference_time, compute_confusion_matrix
from utils import load_checkpoint, count_parameters

import matplotlib.pyplot as plt
import seaborn as sns


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 128

    _, test_loader = get_cifar10_dataloaders(DATA_DIR, batch_size)


    teacher_model = TeacherModel().to(device)
    teacher_model = load_checkpoint(teacher_model, TEACHER_CHECKPOINT_PATH, device)
    teacher_model.eval()

    """
    teacher_model.requires_grad_(False)
    distilled_student_model = StudentModel().to(device)
    distilled_student_model = load_checkpoint(distilled_student_model, distill_student_checkpoint_path, device)
    distilled_student_model.eval() """

    teacher_acc = compute_accuracy(teacher_model, test_loader, device)
    # distilled_acc = compute_accuracy(distilled_student_model, test_loader, device)

    teacher_params = count_parameters(teacher_model)
    # distilled_params = count_parameters(distilled_student_model)

    teacher_time = measure_inference_time(teacher_model, test_loader, device)
    # distilled_time = measure_inference_time(distilled_student_model, test_loader, device)

    """
    results = pd.DataFrame({"Model": ["Baseline Student", "Teacher", "Distilled Student"],
                            "Accuracy": [student_acc, teacher_acc, distilled_acc],
                            "Params": [student_params, teacher_params, distilled_params],
                            "Inference Time": [student_time, teacher_time, distilled_time]}) """
    results = pd.DataFrame({"Model": ["Teacher"],
                            "Accuracy": [teacher_acc],
                            "Params": [teacher_params],
                            "Inference Time": [teacher_time]})
    print("\n===== MODEL COMPARISON =====")
    print(
        results)
    print("\nComputing confusion matrices...")

    # compute_confusion_matrix( distilled_student_model, test_loader, device)
    cm = compute_confusion_matrix( teacher_model, test_loader, device)

    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Teacher")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
