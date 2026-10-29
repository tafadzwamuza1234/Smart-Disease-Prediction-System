# Smart Disease Prediction System Using Ensemble Machine Learning

A university machine learning project that predicts the most likely disease
from a set of reported symptoms, using **ensemble machine learning**
(several classical ML models combined together) rather than a single model.

---

## 1. Project Title

**Smart Disease Prediction System Using Ensemble Machine Learning**

## 2. Project Objective

To design, build, evaluate, and deploy a disease-classification system
that:
- takes a set of patient-reported symptoms as input,
- predicts the most likely disease using **multiple machine learning
  models combined through ensemble learning** (Hard Voting, Soft Voting,
  and Stacking), and
- exposes the result through a simple web application, with an explicit
  medical disclaimer, for a university demonstration.

## 3. Problem Statement

Diagnosing a disease from symptoms alone is a hard, high-stakes,
multi-class classification problem: the same symptom (e.g. "fatigue" or
"headache") can point to dozens of different conditions, and no single
symptom is decisive on its own. This project frames the task as
**supervised multi-class classification**: given a fixed vocabulary of
possible symptoms, predict which one disease (out of many) a patient is
most likely to have. Because no single classical ML algorithm is
uniformly best at this kind of problem, the project's core idea is to
**combine several complementary models with ensemble learning** to get a
more robust prediction than any single model could give alone.

## 4. Dataset Description

- **Source**: a "Symptoms to Diseases" dataset (`data/dataset.csv`),
  246,945 patient records.
- **Features**: 377 columns, one per possible symptom, each a binary
  indicator (`1` = symptom present, `0` = symptom absent).
- **Target**: `diseases` — a text label naming the diagnosed disease.
- **Raw scale**: 713 unique diseases, with an extremely skewed
  distribution — from 1,219 records for the most common disease down to
  just 1 record for the rarest.
- **Data quality**: no missing values anywhere, but 57,305 exact
  duplicate rows.

### A documented scoping decision

Ensemble techniques (Soft Voting, Stacking), Gradient Boosting, k-fold
cross-validation, and hyperparameter search each **multiply** training
time by (re-)training several models several times over. Doing this
honestly across all 713 raw disease classes — many with fewer than 10
examples, some with just 1 — was tested during development and found to
be **impractical on typical student hardware** (this project was
developed/tested on a single CPU core with ~4GB RAM), and statistically
unreliable for the rare classes regardless of hardware: a model cannot
learn a class it barely ever sees, and a "macro F1-score" computed over a
1-example class is close to meaningless.

**Decision**: after removing duplicates, the project scopes the
supervised-learning task to the **top 40 most frequent diseases**
(47,246 records, each disease has 832–1,219 examples — a mild, easily
handled imbalance). The complete raw 713-disease dataset is still shipped
in `data/dataset.csv` so the scope can be widened later with more compute.
This trade-off is explained in detail, with the exact numbers, in
`notebooks/disease_prediction.ipynb` (Stage 2).

## 5. Technologies Used

- Python 3.12
- pandas, numpy — data loading & manipulation
- scikit-learn — models, ensembling, cross-validation, tuning, metrics
- matplotlib, seaborn — visualizations
- joblib — model serialization
- Streamlit — web application
- Jupyter (nbformat/nbclient) — the project notebook

## 6. Machine Learning Models Used

Six baseline classifiers, each trained and evaluated independently:

| Model | Notes |
|---|---|
| Logistic Regression | linear baseline, `class_weight="balanced"` |
| Decision Tree | `max_depth=25`, illustrates high variance of a single tree |
| Random Forest | 150 trees, `max_depth=25` |
| SVM | linear kernel, `probability=True` (needed for Soft Voting) |
| K-Nearest Neighbors | `n_neighbors=7` |
| Gradient Boosting | `HistGradientBoostingClassifier` — see note below |

> **Why `HistGradientBoostingClassifier` instead of classic
> `GradientBoostingClassifier`?** Classic gradient boosting was
> benchmarked on this 40-class problem and found to be dramatically
> slower (a single fit exceeded several minutes even at very low
> iteration counts, because it must effectively build one tree per class
> per boosting round). `HistGradientBoostingClassifier` is scikit-learn's
> modern, histogram-based re-implementation of the same algorithm family
> — same conceptual technique, far more practical at this scale. This
> substitution is documented, not hidden.

## 7. Ensemble Learning Approach

Three ensembling strategies, all built on top of the six baseline models:

1. **Hard Voting** — majority vote of predicted class labels across all 6
   baseline models.
2. **Soft Voting** — average of predicted class probabilities across all
   6 baseline models, then arg-max.
3. **Stacking** — a meta-classifier (Logistic Regression) trained on the
   predicted probabilities of 4 base learners (Logistic Regression,
   Random Forest, linear SVM, Gradient Boosting). To keep this
   computationally practical, the meta-features are generated using a
   **hold-out split** of the training data rather than full 5-fold
   `cross_val_predict` — this still fully avoids data leakage (the
   meta-model never sees a prediction made by a model that was trained on
   that very data point) while costing roughly 2x base-model training
   instead of 5x. See `src/train_models.py::build_stacking_holdout` for
   the exact implementation and reasoning.

Hard/Soft Voting reuse the **already-fitted** baseline models instead of
retraining them a second time (see `ManualVotingEnsemble` in
`src/train_models.py`) — mathematically identical to scikit-learn's
`VotingClassifier`, without the wasted computation.

### An honest result

On the held-out test set, the best single model (linear SVM, macro-F1 =
0.9427) and the Stacking Ensemble (macro-F1 = 0.9426) are **statistically
tied** — the base learners here are already strong and make fairly
similar mistakes, leaving limited "diversity" for an ensemble to exploit.
This is a genuine, useful finding about when ensembling helps a lot
versus only marginally. Because the project's goal is specifically an
ensemble system, and the difference is not meaningful, **the Stacking
Ensemble is the model that gets deployed** in the Streamlit app — this
decision is made transparently, alongside the full comparison table, in
the notebook (Stage 11).

## 8. Data Preprocessing

1. Loaded the raw CSV exactly as provided (no assumed column names).
2. Removed 57,305 exact duplicate rows.
3. Scoped the disease list to the top 40 most frequent diseases (see
   Section 4).
4. No further encoding was required — symptom features are already
   binary 0/1 indicators.
5. Stratified 80/20 train/test split (`random_state=42`), so every
   disease is proportionally represented in both sets.
6. Class imbalance (mild, ~1.5x between largest/smallest of the 40 kept
   diseases) handled with `class_weight="balanced"` on the models that
   support it, rather than synthetic oversampling.

## 9. Model Evaluation

All models are evaluated on the same held-out test set (9,450 records)
using accuracy, precision (macro), recall (macro), F1-score, macro
F1-score, and weighted F1-score. **Macro-F1 is the primary metric** used
to rank models, since it treats every disease equally regardless of how
common it is — a model that quietly ignores rarer diseases while acing
the common ones would still look good on accuracy alone, but not on
macro-F1.

**Full comparison table** (from `results/model_comparison.csv`, produced
by an actual run of this project — not hand-typed):

| Model | Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| SVM (linear kernel) | 0.9415 | 0.9438 | 0.9428 | **0.9427** | 0.9416 |
| Stacking Ensemble | 0.9415 | 0.9447 | 0.9426 | 0.9426 | 0.9415 |
| Logistic Regression (tuned) | 0.9415 | 0.9436 | 0.9429 | 0.9420 | 0.9408 |
| Logistic Regression | 0.9402 | 0.9420 | 0.9416 | 0.9411 | 0.9399 |
| K-Nearest Neighbors | 0.9399 | 0.9412 | 0.9410 | 0.9410 | 0.9398 |
| Random Forest (tuned) | 0.9400 | 0.9417 | 0.9411 | 0.9409 | 0.9399 |
| Soft Voting Ensemble | 0.9394 | 0.9409 | 0.9405 | 0.9405 | 0.9394 |
| Hard Voting Ensemble | 0.9390 | 0.9407 | 0.9403 | 0.9403 | 0.9391 |
| Gradient Boosting | 0.9368 | 0.9383 | 0.9379 | 0.9380 | 0.9368 |
| Random Forest | 0.9362 | 0.9398 | 0.9373 | 0.9378 | 0.9367 |
| Decision Tree | 0.3486 | 0.6498 | 0.3493 | 0.4340 | 0.4326 |

*(The Decision Tree's low score at this depth setting is a genuine,
useful illustration of why a single tree is a weak, high-variance model
compared to an ensemble of trees — see the notebook for discussion.)*

**3-fold stratified cross-validation** on the training set (from
`results/cross_validation_results.csv`) confirms these numbers are not a
fluke of one particular split — e.g. SVM: macro-F1 = 0.946 ± 0.001,
Logistic Regression: 0.946 ± 0.001, Random Forest: 0.942 ± 0.001,
Gradient Boosting: 0.941 ± 0.002, Decision Tree: 0.435 ± 0.012.

**Hyperparameter tuning** (`RandomizedSearchCV`, 3-fold CV) was applied to
Random Forest and Logistic Regression (chosen because they are fast to
refit many times, and both are Stacking base learners). Best Random
Forest parameters: `n_estimators=300, max_depth=20, min_samples_leaf=4,
max_features='log2'`. Best Logistic Regression parameters: `C=0.01,
solver='lbfgs'`.

## 10. How to Train the Models

```bash
cd smart-disease-prediction
pip install -r requirements.txt

# Run the full pipeline (checkpointed — safe to re-run / resume):
python src/run_pipeline.py all

# Or run one stage at a time (recommended on limited-RAM machines,
# since each stage then runs in its own fresh process):
python src/run_pipeline.py data
python src/run_pipeline.py baselines
python src/run_pipeline.py ensembles
python src/run_pipeline.py cv
python src/run_pipeline.py tuning
python src/run_pipeline.py final
```

Every stage saves a checkpoint to `results/checkpoints/`, so re-running a
stage that has already completed simply loads its saved result instead of
re-training. Delete a specific `.joblib` file in that folder to force that
stage to recompute.

Alternatively, open `notebooks/disease_prediction.ipynb` and run all
cells — it imports and calls the exact same functions.

## 11. How to Run the Streamlit Application

```bash
cd smart-disease-prediction
pip install -r requirements.txt
streamlit run app.py
```

The app only loads the saved `models/ensemble_model.pkl` bundle — it
never retrains anything, so it starts instantly. If that file does not
exist yet, run the training pipeline first (Section 10).

## 12. Example Prediction

Using the saved pipeline (`src/predict.py`):

```python
symptoms = ["headache", "nausea", "dizziness"]
# -> predicted_disease: "concussion"
#    confidence: 0.935
#    top_3: concussion (0.935), noninfectious gastroenteritis (0.003),
#           infectious gastroenteritis (0.003)
```

```python
symptoms = ["sore_throat", "cough", "fever", "this_symptom_does_not_exist"]
# -> unrecognized_symptoms: ["this_symptom_does_not_exist"] (safely ignored)
#    predicted_disease: "pneumonia"
#    confidence: 0.307
#    top_3: pneumonia (0.307), acute bronchitis (0.225), strep throat (0.195)
```

Both examples are real outputs from an actual run — see
`notebooks/disease_prediction.ipynb`, Stage 13.

## 13. Limitations

- **Scope**: only the 40 most frequent diseases (of 713 in the raw data)
  are modeled — see Section 4 for the reasoning.
- **Symptom-only input**: the model has no access to patient history,
  age, lab results, imaging, or vital signs — only a checklist of
  symptoms, which is a major simplification of real diagnosis.
- **Dataset realism**: like many public symptom-disease datasets, the
  data may not perfectly reflect real-world symptom co-occurrence or
  reporting bias.
- **Compute-driven choices**: several settings (3-fold rather than
  5-fold CV, `HistGradientBoostingClassifier` instead of classic GB, a
  hold-out rather than k-fold stacking scheme, tuning only 2 of the 6
  models) were chosen to keep the project practical on limited student
  hardware, and are documented wherever they appear.
- **Confidence ≠ certainty**: the reported probability is the model's
  internal confidence, not a validated clinical probability.

## 14. Medical Disclaimer

**This system is an educational / research machine learning project
developed for a university course. It is not a medical diagnosis and
must not be used as a substitute for professional medical advice,
diagnosis, or treatment. Always consult a qualified healthcare
professional with any questions regarding a medical condition.** This
disclaimer is also shown directly in the Streamlit application.

---

## Project Structure

```
smart-disease-prediction/
│
├── data/
│   ├── dataset.csv                  # full raw dataset (713 diseases, 246,945 rows)
│   └── symptom_text_reference.csv   # supplementary symptom/disease text reference
│
├── notebooks/
│   └── disease_prediction.ipynb     # full, executed, end-to-end notebook (Stages 1-13 + viva guide)
│
├── src/
│   ├── preprocessing.py             # Stages 1-5: load, clean, scope, split
│   ├── eda.py                       # Stage 3: plotting helpers
│   ├── train_models.py              # Stages 6-12: baselines, ensembles, CV, tuning, saving
│   ├── evaluate_models.py           # comparison table, confusion matrix, feature importance
│   ├── run_pipeline.py              # checkpointed CLI runner for the whole pipeline
│   └── predict.py                   # Stage 13: prediction pipeline for new users
│
├── models/
│   └── ensemble_model.pkl           # final deployed model + feature names + class labels
│
├── results/                         # metrics tables, dataset/cleaning reports, figures
│   └── figures/
│
├── app.py                           # Stage 14: Streamlit web application
├── requirements.txt
├── README.md
└── .gitignore
```

## Reproducibility

`random_state=42` is fixed everywhere (train/test split, all models,
cross-validation folds, hyperparameter search). Developed and tested with
Python 3.12 and the package versions in `requirements.txt`.

---

## How to Explain This Project in a Viva

1. **Data**: real symptoms-to-disease dataset, ~247,000 records, 377
   possible symptoms, originally 713 diseases; cleaned (duplicates
   removed) and scoped to the 40 most frequent, best-represented
   diseases for a statistically reliable, hardware-practical project.
2. **Features & target**: each record is a 0/1 vector over 377 symptoms
   (features) and one disease name (target).
3. **Split**: stratified 80/20 train/test split so disease proportions
   stay balanced in both sets.
4. **Baseline models**: six classical ML models trained and compared on
   the same test set.
5. **Ensemble learning**: combined the six models three ways — Hard
   Voting, Soft Voting, and Stacking (a trained meta-model) — because
   different models make different mistakes, and combining them can be
   more robust than trusting any one of them.
6. **Reliable evaluation**: k-fold cross-validation confirms the
   train/test split results aren't a fluke; macro-F1 (not just accuracy)
   ensures rarer diseases aren't ignored in the evaluation.
7. **Tuning**: `RandomizedSearchCV` on two of the models, validated only
   on the training data (never the test set) to avoid data leakage.
8. **Final model**: compared every model on macro-F1 first; found the
   best single model and best ensemble were statistically tied, and
   deployed the Stacking Ensemble since the project's goal is
   specifically an ensemble system.
9. **Prediction pipeline**: converts a user's symptom list into the same
   0/1 feature format used in training, predicts with the saved model,
   and returns the top-3 most likely diseases with probabilities —
   gracefully ignoring any symptom it doesn't recognize.
10. **Streamlit app**: a simple web interface around the same pipeline,
    with a clear medical disclaimer that this is an educational project,
    not a medical diagnosis.
