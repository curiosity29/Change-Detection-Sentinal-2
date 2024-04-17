from functools import partial
from torcheval.metrics.functional.classification import multiclass_recall
from torcheval.metrics.functional import multiclass_precision, binary_precision
from torcheval.metrics.functional.classification import binary_recall
import torch

def get_metrics_calculator(n_class):
    if n_class == 1:
        return {
        "precision": binary_precision_calculator,
        "recall": binary_recall_calculator,
        }
    return {
        "precision": partial(precision_calculator, n_class = n_class),
        "recall": partial(recall_calculator, n_class = n_class),
    }

def binary_recall_calculator(y_true, y_pred):
    return binary_recall(input = y_pred.flatten(), target = y_true.flatten().int())

def binary_precision_calculator(y_true, y_pred):
    return binary_precision(input = y_pred.flatten(), target = y_true.flatten().int())

    
def recall_calculator(y_true, y_pred, n_class):
    y_true = torch.argmax(y_true).flatten().int()
    y_pred = torch.argmax(y_pred).flatten()
    return multiclass_recall(input = y_pred, target = y_true, average="macro", num_classes=n_class)

def precision_calculator(y_true, y_pred, n_class):
    y_true = torch.argmax(y_true).flatten().int()
    y_pred = torch.argmax(y_pred).flatten()
    return multiclass_precision(input = y_pred, target = y_true, average="macro", num_classes=n_class)

