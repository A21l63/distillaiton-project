import torch
import pandas as pd
from data import get_cifar10_dataloaders
from models import StudentModel, TeacherModel
from evaluate import compute_accuracy, measure_inference_time, compute_confusion_matrix
from utils import load_checkpoint, count_parameters
#from main_distill_student import teacher_checkpoint_path
#from main_distill_student import distill_student_checkpoint_path



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = "./cifar-10"
    batch_size = 128
    num_epochs = 10
    learning_rate = 0.01
    student_checkpoint_path = "./student_model"

    train_loader, test_loader = get_cifar10_dataloaders(data_dir, batch_size)

    student_model = StudentModel().to(device)
    student_model = load_checkpoint(student_model, student_checkpoint_path, device)
    student_model.eval()

    '''teacher_model = TeacherModel().to(device)
    teacher_model = load_checkpoint(teacher_model, teacher_checkpoint_path, device)
    teacher_model.eval()'''

    """
    teacher_model.requires_grad_(False)
    distilled_student_model = StudentModel().to(device)
    distilled_student_model = load_checkpoint(distilled_student_model, distill_student_checkpoint_path, device)
    distilled_student_model.eval() """


    student_acc = compute_accuracy(student_model, test_loader, device)
    #teacher_acc = compute_accuracy(teacher_model, test_loader, device)
    # distilled_acc = compute_accuracy(distilled_student_model, test_loader, device)

    student_params = count_parameters(student_model)
    #teacher_params = count_parameters(teacher_model)
    # distilled_params = count_parameters(distilled_student_model)

    student_time = measure_inference_time(student_model, test_loader, device)
    #teacher_time = measure_inference_time(teacher_model, test_loader, device)
    # distilled_time = measure_inference_time(distilled_student_model, test_loader, device)

    """
    results = pd.DataFrame({"Model": ["Baseline Student", "Teacher", "Distilled Student"],
                            "Accuracy": [student_acc, teacher_acc, distilled_acc],
                            "Params": [student_params, teacher_params, distilled_params],
                            "Inference Time": [student_time, teacher_time, distilled_time]}) """
    results = pd.DataFrame({"Model": ["Baseline Student"],
                            "Accuracy": [student_acc,],
                            "Params": [student_params],
                            "Inference Time": [student_time]})
    print("\n===== MODEL COMPARISON =====")
    print(
        results)
    print("\nComputing confusion matrices...")


    cm = compute_confusion_matrix( student_model, test_loader, device, title="Baseline Student" )
    print(cm)
    # compute_confusion_matrix( distilled_student_model, test_loader, device, title="Distilled Student" )
    #compute_confusion_matrix( teacher_model, test_loader, device, title="Teacher" )


if __name__ == "__main__":
    main()
