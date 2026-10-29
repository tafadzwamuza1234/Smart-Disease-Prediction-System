"""
train_models.py
----------------
Stages 6-12 of the Smart Disease Prediction System:
    6.  Individual baseline models
    7.  Model evaluation
    8.  Ensemble learning (hard voting, soft voting, stacking)
    9.  Cross-validation
    10. Hyperparameter tuning
    11. Final model selection
    12. Saving the final model + preprocessing objects

This file can be run directly:
    python src/train_models.py
and it will reproduce every step reported in the notebook, because the
notebook imports these exact same functions -- there is only one
implementation of the machine learning logic.
"""

import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.dirname(__file__))
import preprocessing as prep  # noqa: E402

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Stage 6: Baseline models
# ---------------------------------------------------------------------------
def get_baseline_models():
    """
    Returns a dictionary of FRESH (unfitted) baseline classifiers.

    Notes on choices made for a resource-constrained student machine:

    * SVM -> SVC(kernel="linear", probability=True). A linear kernel is
      used (instead of RBF) because the input space already has 377
      binary symptom features, which is enough for a linear decision
      boundary to work well, and it keeps training time reasonable for
      a 40-class problem. probability=True is enabled so the SVM can
      participate in Soft Voting, as requested.

    * Gradient Boosting -> HistGradientBoostingClassifier. This is
      scikit-learn's histogram-based, modern re-implementation of
      gradient boosting (conceptually the same algorithm family as
      classic GradientBoostingClassifier, but dramatically faster and
      far more memory-efficient on larger / multi-class datasets --
      classic GradientBoostingClassifier was measured to be
      impractically slow for a 40+ class problem on a single CPU core).
      This substitution is documented here and in the README.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=300, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=25, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=25,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "SVM (linear kernel)": SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7, n_jobs=1),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=60, random_state=RANDOM_STATE
        ),
    }
    return models


# ---------------------------------------------------------------------------
# Stage 7: Evaluation
# ---------------------------------------------------------------------------
def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def fit_and_evaluate(name, clf, X_train, y_train, X_test, y_test, verbose=True):
    t0 = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - t0

    y_pred = clf.predict(X_test)
    metrics = evaluate_predictions(y_test, y_pred)
    metrics["model"] = name
    metrics["fit_time_seconds"] = round(fit_time, 2)

    if verbose:
        print(
            f"[{name}] fit_time={fit_time:6.2f}s  "
            f"accuracy={metrics['accuracy']:.3f}  "
            f"f1_macro={metrics['f1_macro']:.3f}  "
            f"f1_weighted={metrics['f1_weighted']:.3f}"
        )
    return clf, metrics, y_pred


def train_all_baselines(models, X_train, y_train, X_test, y_test):
    fitted = {}
    metrics_list = []
    predictions = {}
    for name, clf in models.items():
        fitted_clf, metrics, y_pred = fit_and_evaluate(
            name, clf, X_train, y_train, X_test, y_test
        )
        fitted[name] = fitted_clf
        metrics_list.append(metrics)
        predictions[name] = y_pred
    metrics_df = pd.DataFrame(metrics_list).set_index("model")
    return fitted, metrics_df, predictions


# ---------------------------------------------------------------------------
# Stage 8: Ensemble learning
# ---------------------------------------------------------------------------
class ManualVotingEnsemble(BaseEstimator, ClassifierMixin):
    """
    A lightweight Hard/Soft Voting ensemble that REUSES already-fitted
    base estimators instead of retraining them (sklearn's
    VotingClassifier always clones and refits its estimators, which
    would mean training every base model a second time for no benefit,
    since they are already fitted on the exact same training data).

    voting="hard" -> majority vote of the predicted class labels.
    voting="soft" -> average of predict_proba across base estimators,
                     then argmax. Requires every base estimator to
                     support predict_proba.
    """

    def __init__(self, fitted_estimators: dict, voting="soft"):
        assert voting in ("hard", "soft")
        self.fitted_estimators = fitted_estimators
        self.voting = voting
        first_model = next(iter(fitted_estimators.values()))
        self.classes_ = first_model.classes_

    def fit(self, X, y):
        # No-op: this ensemble is built from already-fitted estimators.
        # This method only exists so scikit-learn utilities that check
        # for a `fit` method (e.g. permutation_importance) accept this
        # object as a valid "estimator".
        return self

    def predict(self, X):
        if self.voting == "hard":
            preds = np.array([clf.predict(X) for clf in self.fitted_estimators.values()])
            # majority vote per sample (works for string class labels too,
            # unlike scipy.stats.mode which now requires numeric input)
            n_samples = preds.shape[1]
            winners = np.empty(n_samples, dtype=preds.dtype)
            for i in range(n_samples):
                values, counts = np.unique(preds[:, i], return_counts=True)
                winners[i] = values[np.argmax(counts)]
            return winners
        else:
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        probas = [clf.predict_proba(X) for clf in self.fitted_estimators.values()]
        return np.mean(probas, axis=0)


def build_stacking_holdout(
    base_model_factory: dict,
    X_train,
    y_train,
    meta_model,
    holdout_frac=0.3,
    random_state=RANDOM_STATE,
):
    """
    Stage 8: Stacking ensemble using a HOLD-OUT split (instead of full
    k-fold cross_val_predict) to generate the meta-features.

    Why a hold-out split instead of the "textbook" 5-fold
    cross_val_predict approach?  With 6 base models this project
    already spends a meaningful amount of compute time on a
    single-core machine; generating out-of-fold predictions with
    5-fold CV would mean fitting every stacking base-model FIVE times.
    A single hold-out split still fully avoids data leakage (the
    meta-model never sees predictions made by a model that was trained
    on that same data) while costing roughly the same as training the
    base models twice. This is a documented, deliberate trade-off for
    a resource-constrained environment.

    Steps:
      1. Split X_train/y_train into train_base and train_meta.
      2. Fit a *fresh* copy of every base model on train_base only.
      3. Use those models' predict_proba on train_meta to build the
         meta-feature matrix (this is what the base models predict on
         data they were NOT trained on -- no leakage).
      4. Train the meta-model on (meta-features, y_train_meta).
      5. Re-fit fresh copies of every base model on the FULL X_train,
         so that at prediction time the base models benefit from all
         available training data.

    Returns a small StackingEnsemble object exposing predict / predict_proba.
    """
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split

    X_base, X_meta, y_base, y_meta = train_test_split(
        X_train, y_train, test_size=holdout_frac, random_state=random_state, stratify=y_train
    )

    # Step 2-3: train base models on train_base, predict on train_meta
    holdout_fitted = {}
    meta_features_list = []
    class_order = None
    for name, factory in base_model_factory.items():
        clf = clone(factory)
        clf.fit(X_base, y_base)
        proba = clf.predict_proba(X_meta)
        if class_order is None:
            class_order = clf.classes_
        else:
            # align columns defensively (should already match since all
            # base models are fit on the same y_base classes)
            assert list(class_order) == list(clf.classes_)
        meta_features_list.append(proba)
        holdout_fitted[name] = clf

    meta_X = np.hstack(meta_features_list)
    meta_model = clone(meta_model)
    meta_model.fit(meta_X, y_meta)

    # Step 5: refit fresh base models on the FULL training data for
    # deployment / final inference.
    final_base_models = {}
    for name, factory in base_model_factory.items():
        clf = clone(factory)
        clf.fit(X_train, y_train)
        final_base_models[name] = clf

    return StackingEnsemble(final_base_models, meta_model, class_order)


class StackingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models: dict, meta_model, class_order):
        self.base_models = base_models
        self.meta_model = meta_model
        self.class_order = class_order
        self.classes_ = meta_model.classes_

    def fit(self, X, y):
        # No-op: see ManualVotingEnsemble.fit for the reason this exists.
        return self

    def _meta_features(self, X):
        feats = []
        for clf in self.base_models.values():
            feats.append(clf.predict_proba(X))
        return np.hstack(feats)

    def predict(self, X):
        return self.meta_model.predict(self._meta_features(X))

    def predict_proba(self, X):
        return self.meta_model.predict_proba(self._meta_features(X))


# ---------------------------------------------------------------------------
# Stage 9: Cross-validation
# ---------------------------------------------------------------------------
def cross_validate_models(model_factory: dict, X_train, y_train, cv_folds=3):
    """
    Uses StratifiedKFold so every fold has a representative share of
    every disease class. cv_folds=3 (rather than the more common 5) is
    chosen because some diseases have relatively few training examples
    after the train/test split, and because of single-core compute
    constraints -- 3-fold CV already multiplies total training time by 3.
    """
    from sklearn.base import clone

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, factory in model_factory.items():
        t0 = time.time()
        scores = cross_val_score(
            clone(factory), X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=1
        )
        results[name] = {
            "cv_f1_macro_mean": float(scores.mean()),
            "cv_f1_macro_std": float(scores.std()),
            "cv_time_seconds": round(time.time() - t0, 2),
        }
        print(
            f"[CV] {name}: f1_macro = {scores.mean():.3f} +/- {scores.std():.3f} "
            f"({time.time() - t0:.1f}s)"
        )
    return pd.DataFrame(results).T


# ---------------------------------------------------------------------------
# Stage 10: Hyperparameter tuning
# ---------------------------------------------------------------------------
def tune_random_forest(X_train, y_train, n_iter=6, cv_folds=3):
    """
    We focus tuning effort on Random Forest (a bagging ensemble member)
    and Logistic Regression (the linear baseline) rather than every
    model, because:
      * SVM and Gradient Boosting are already the slowest models to
        fit once -- an exhaustive search over them would dominate the
        whole project's runtime on a single CPU core.
      * Random Forest and Logistic Regression are fast enough that a
        RandomizedSearchCV with a handful of candidates is genuinely
        practical, and both are used as stacking base learners, so
        improving them also improves the final ensemble.
    """
    param_dist = {
        "n_estimators": [100, 150, 200, 300],
        "max_depth": [15, 20, 25, 30, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    base = RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1_macro",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    print(f"[Tuning] Random Forest best f1_macro (CV)={search.best_score_:.3f} "
          f"in {time.time()-t0:.1f}s | best_params={search.best_params_}")
    return search


def tune_logistic_regression(X_train, y_train, n_iter=5, cv_folds=3):
    param_dist = {
        "C": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
        "solver": ["lbfgs"],
    }
    base = LogisticRegression(
        max_iter=300, class_weight="balanced", random_state=RANDOM_STATE
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE),
        scoring="f1_macro",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    print(f"[Tuning] Logistic Regression best f1_macro (CV)={search.best_score_:.3f} "
          f"in {time.time()-t0:.1f}s | best_params={search.best_params_}")
    return search


# ---------------------------------------------------------------------------
# Stage 11-12: Final model selection + saving
# ---------------------------------------------------------------------------
def save_final_artifacts(final_model, feature_names, class_labels, models_dir,
                          extra_objects=None):
    os.makedirs(models_dir, exist_ok=True)
    bundle = {
        "model": final_model,
        "feature_names": feature_names,
        "class_labels": list(class_labels),
    }
    if extra_objects:
        bundle.update(extra_objects)
    path = os.path.join(models_dir, "ensemble_model.pkl")
    joblib.dump(bundle, path)
    print(f"Saved final model bundle to {path}")
    return path
