import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
import torch

def predict(model, dataloader, device) -> tuple[list, list, list]:
    """


    Parameters
    ----------


    Returns
    -------

    """
    all_labels = []
    all_preds = []
    all_confidence = []

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(X)
            prob = torch.softmax(logits, dim=1)
            confidence, preds = prob.max(dim=1)

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_confidence.extend(confidence.cpu().numpy())

    return all_labels, all_preds, all_confidence

def compute_metrics(y_pred, y_true, class_names, report_f1: bool = False):
    """


    Parameters
    ----------


    Returns
    -------

    """
    report = classification_report(y_true, y_pred, target_names=class_names)

    raw_cm = confusion_matrix(y_true, y_pred)
    row_cm = confusion_matrix(y_true, y_pred, normalize='true')
    column_cm = confusion_matrix(y_true, y_pred, normalize='pred')

    confusion_matrices = {'raw': raw_cm, 'row': row_cm, 'column': column_cm}

    data = []

    for i, class_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            if i != j:
                data.append({'class': class_name, 'pred': pred_name, 'error': row_cm[i, j]})

    class_confusion = pd.DataFrame(data=data)
    class_confusion["Count"] = [raw_cm[i, j] for i in range(len(class_names))
                                for j in range(len(class_names)) if i != j]
    class_confusion.sort_values(by='Count', ascending=False, inplace=True, ignore_index=True)

    accuracy = accuracy_score(y_true, y_pred)

    if report_f1:
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        return report, confusion_matrices, class_confusion, accuracy, macro_f1, weighted_f1

    return report, confusion_matrices, class_confusion, accuracy

