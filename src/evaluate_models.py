"""
evaluate_models.py
-------------------
Helper functions used after models are trained: building the final
comparison table, generating confusion matrices, and computing feature
importance for the final model. Kept separate from train_models.py so
evaluation/reporting can be re-run without re-training anything, by
loading the saved artifacts in models/ and results/.
"""

import os

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix


def build_comparison_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces the final Model / Accuracy / Precision / Recall / F1-score /
    Macro F1-score / Weighted F1-score comparison table requested for
    the project report.
    """
    table = metrics_df.copy()
    table = table.rename(
        columns={
            "accuracy": "Accuracy",
            "precision_macro": "Precision (macro)",
            "recall_macro": "Recall (macro)",
            "f1_macro": "Macro F1-score",
            "f1_weighted": "Weighted F1-score",
        }
    )
    # A single "F1-score" column (weighted) is included for readability,
    # matching how it is usually reported for imbalanced multi-class
    # problems, while macro/weighted are kept explicit as requested.
    table["F1-score"] = table["Weighted F1-score"]
    ordered_cols = [
        "Accuracy",
        "Precision (macro)",
        "Recall (macro)",
        "F1-score",
        "Macro F1-score",
        "Weighted F1-score",
    ]
    ordered_cols = [c for c in ordered_cols if c in table.columns] + [
        c for c in table.columns if c not in ordered_cols
    ]
    return table[ordered_cols].sort_values("Macro F1-score", ascending=False)


def get_classification_report_dict(y_true, y_pred):
    return classification_report(y_true, y_pred, zero_division=0, output_dict=True)


def get_confusion_matrix(y_true, y_pred, labels):
    return confusion_matrix(y_true, y_pred, labels=labels)


def top_confused_pairs(cm, labels, top_k=10):
    """
    Returns the (true, predicted, count) triples with the most
    off-diagonal (i.e. incorrect) confusions -- useful to explain, in a
    viva, which diseases the model tends to mix up and why (usually
    diseases that share many overlapping symptoms).
    """
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((labels[i], labels[j], int(cm[i, j])))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs[:top_k]


def permutation_feature_importance(model, X_val, y_val, feature_names, n_repeats=5,
                                    random_state=42, n_top=25):
    """
    Used when the final model does not directly expose
    feature_importances_ (e.g. an ensemble / stacking model). Measures
    how much the macro-F1 score drops when a single symptom column is
    randomly shuffled -- a bigger drop means that symptom matters more
    for correct predictions.
    """
    result = permutation_importance(
        model, X_val, y_val, n_repeats=n_repeats, random_state=random_state,
        scoring="f1_macro", n_jobs=1,
    )
    importances = pd.Series(result.importances_mean, index=feature_names)
    return importances.sort_values(ascending=False).head(n_top)
