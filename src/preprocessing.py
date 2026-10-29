"""
preprocessing.py
-----------------
Stage 1-5 helper functions for the Smart Disease Prediction System.

This module is responsible for:
    * loading the raw "Symptoms to Diseases" dataset
    * producing a full dataset report (shape, dtypes, missing values,
      duplicates, class distribution, basic statistics)
    * cleaning the data (removing duplicate rows, scoping the disease
      list to a computationally-tractable, statistically reliable set)
    * separating features (X) from the target (y)
    * creating a reproducible, stratified train/test split

All functions are written so that both the Jupyter notebook and the
command-line scripts (train_models.py, evaluate_models.py, predict.py)
can import and reuse the exact same logic -- there is only ONE place
where preprocessing rules live.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TARGET_COLUMN = "diseases"

# The raw dataset contains 713 disease classes. Many of them have fewer than
# 10 examples (some have only 1), which makes it impossible to even perform a
# stratified train/test split for those classes, let alone learn a reliable
# decision boundary for them. To keep the ensemble-learning problem
# statistically meaningful AND computationally tractable on a normal
# student laptop (this project was developed/tested on a single-core,
# ~4GB RAM machine), we scope the supervised learning task to the
# TOP_N_DISEASES most frequent diseases. The full raw dataset is still
# shipped in data/dataset.csv so the scope can be widened later if more
# compute is available.
TOP_N_DISEASES = 40

TEST_SIZE = 0.2


def load_raw_data(path):
    """Stage 1: Load the raw CSV exactly as provided, no assumptions."""
    df = pd.read_csv(path)
    return df


def dataset_report(df, target_column=TARGET_COLUMN):
    """
    Stage 1: Produce a complete, honest description of the dataset.
    Returns a dictionary so it can be reused in the notebook, printed,
    or serialized to JSON for the README.
    """
    report = {}
    report["shape"] = df.shape
    report["n_rows"] = int(df.shape[0])
    report["n_columns"] = int(df.shape[1])
    report["target_column"] = target_column
    report["feature_columns_count"] = int(df.shape[1] - 1)
    report["dtypes"] = df.dtypes.astype(str).value_counts().to_dict()
    report["missing_values_total"] = int(df.isnull().sum().sum())
    report["duplicate_rows"] = int(df.duplicated().sum())
    report["n_unique_diseases"] = int(df[target_column].nunique())

    class_counts = df[target_column].value_counts()
    report["class_count_min"] = int(class_counts.min())
    report["class_count_max"] = int(class_counts.max())
    report["class_count_mean"] = float(class_counts.mean())
    report["classes_with_fewer_than_10_examples"] = int((class_counts < 10).sum())

    # Feature columns are all binary (0/1) symptom indicators.
    feature_cols = [c for c in df.columns if c != target_column]
    unique_vals = pd.unique(df[feature_cols].values.ravel())
    report["feature_value_set"] = sorted([int(v) for v in unique_vals])
    report["symptom_prevalence_describe"] = (
        df[feature_cols].sum(axis=0).describe().to_dict()
    )
    return report


def clean_data(df, top_n=TOP_N_DISEASES, target_column=TARGET_COLUMN):
    """
    Stage 2: Clean the dataset.

    Steps:
      1. Drop exact duplicate rows. A duplicated row means the exact same
         combination of symptoms was recorded for the exact same disease
         more than once -- it adds no new information and can bias the
         model towards records that merely happen to be duplicated.
      2. Scope the target to the TOP_N most frequent diseases (see the
         explanation above TOP_N_DISEASES). This is a documented,
         deliberate decision -- not an accident -- and is reported back
         to the caller so it can be shown transparently in the notebook.
    """
    cleaning_log = {}
    cleaning_log["rows_before"] = int(df.shape[0])

    df = df.drop_duplicates().reset_index(drop=True)
    cleaning_log["rows_after_dedup"] = int(df.shape[0])
    cleaning_log["duplicate_rows_removed"] = (
        cleaning_log["rows_before"] - cleaning_log["rows_after_dedup"]
    )

    value_counts = df[target_column].value_counts()
    kept_diseases = value_counts.head(top_n).index.tolist()

    cleaning_log["total_diseases_before_scoping"] = int(value_counts.shape[0])
    cleaning_log["diseases_kept"] = len(kept_diseases)
    cleaning_log["diseases_dropped"] = int(value_counts.shape[0] - len(kept_diseases))

    df_clean = df[df[target_column].isin(kept_diseases)].reset_index(drop=True)
    cleaning_log["rows_after_scoping"] = int(df_clean.shape[0])
    cleaning_log["rows_dropped_by_scoping"] = (
        cleaning_log["rows_after_dedup"] - cleaning_log["rows_after_scoping"]
    )
    cleaning_log["kept_disease_list"] = kept_diseases

    return df_clean, cleaning_log


def get_feature_names(df, target_column=TARGET_COLUMN):
    return [c for c in df.columns if c != target_column]


def split_features_target(df, target_column=TARGET_COLUMN):
    feature_names = get_feature_names(df, target_column)
    X = df[feature_names].copy()
    y = df[target_column].copy()
    return X, y, feature_names


def make_train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Stage 5: Stratified train/test split.

    Stratification matters here because this is a MULTI-CLASS problem
    with some class imbalance (roughly 830-1220 examples per disease
    after scoping) -- a plain random split could easily under- or
    over-represent a disease in the test set, making the evaluation
    metrics misleading.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
