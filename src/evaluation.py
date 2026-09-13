from typing import Dict
import pandas as pd

from .models import compute_macro_metrics


def summarize_model_results(
    name: str, y_train, y_train_pred, y_test, y_test_pred
) -> pd.DataFrame:
    """
    Build a small summary table for a single model.
    """
    train_acc, train_p, train_r, train_f1 = compute_macro_metrics(y_train, y_train_pred)
    test_acc, test_p, test_r, test_f1 = compute_macro_metrics(y_test, y_test_pred)

    data = {
        "model": [name],
        "train_accuracy": [train_acc],
        "train_precision_macro": [train_p],
        "train_recall_macro": [train_r],
        "train_f1_macro": [train_f1],
        "test_accuracy": [test_acc],
        "test_precision_macro": [test_p],
        "test_recall_macro": [test_r],
        "test_f1_macro": [test_f1],
    }
    return pd.DataFrame(data)
