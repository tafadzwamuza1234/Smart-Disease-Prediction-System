"""
eda.py
------
Stage 3: Exploratory Data Analysis plotting helpers.

Each function saves a PNG into results/figures/ AND returns the
matplotlib Figure so it can also be displayed inline in the notebook.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIGSIZE_WIDE = (10, 6)


def _save(fig, out_dir, filename):
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, filename), bbox_inches="tight", dpi=110)


def plot_class_distribution(y, out_dir=None, top_k=40, filename="class_distribution.png"):
    counts = y.value_counts().head(top_k)
    fig, ax = plt.subplots(figsize=(12, 8))
    counts.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("Number of records")
    ax.set_ylabel("Disease")
    ax.set_title(f"Disease (class) distribution - top {top_k} diseases")
    fig.tight_layout()
    _save(fig, out_dir, filename)
    return fig


def plot_symptom_frequency(X, out_dir=None, top_k=25, filename="symptom_frequency.png"):
    freq = X.sum(axis=0).sort_values(ascending=False).head(top_k)
    fig, ax = plt.subplots(figsize=(10, 8))
    freq.sort_values().plot(kind="barh", ax=ax, color="#DD8452")
    ax.set_xlabel("Number of records reporting this symptom")
    ax.set_title(f"Most frequent symptoms (top {top_k})")
    fig.tight_layout()
    _save(fig, out_dir, filename)
    return fig


def plot_symptoms_per_record(X, out_dir=None, filename="symptoms_per_record.png"):
    n_symptoms = X.sum(axis=1)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.hist(n_symptoms, bins=range(0, int(n_symptoms.max()) + 2), color="#55A868", edgecolor="white")
    ax.set_xlabel("Number of symptoms reported in a single record")
    ax.set_ylabel("Number of records")
    ax.set_title("Distribution of symptom-count per record")
    fig.tight_layout()
    _save(fig, out_dir, filename)
    return fig


def plot_model_comparison(comparison_df, metric="f1_macro", out_dir=None,
                           filename="model_comparison.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ordered = comparison_df.sort_values(metric)
    ax.barh(ordered["model"], ordered[metric], color="#8172B2")
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"Model comparison ({metric.replace('_', ' ')})")
    fig.tight_layout()
    _save(fig, out_dir, filename)
    return fig


def plot_confusion_matrix(cm, class_labels, title, out_dir=None, filename="confusion_matrix.png",
                           normalize=True):
    import numpy as np
    import seaborn as sns

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, ax=ax, cmap="Blues", cbar=True, xticklabels=class_labels,
                yticklabels=class_labels, vmin=0, vmax=1 if normalize else None)
    ax.set_xlabel("Predicted disease")
    ax.set_ylabel("Actual disease")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)
    fig.tight_layout()
    _save(fig, out_dir, filename)
    return fig


def plot_feature_importance(importances, feature_names, out_dir=None, top_k=25,
                             filename="feature_importance.png", title="Feature importance"):
    s = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_k)
    fig, ax = plt.subplots(figsize=(10, 8))
    s.sort_values().plot(kind="barh", ax=ax, color="#C44E52")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, out_dir, filename)
    return fig
