import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

try:
    from google.colab import drive
except ImportError:
    drive = None


# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
if drive is not None:
    drive.mount("/content/drive")
    os.chdir("/content/drive/MyDrive/Ramazan Observations /experiments")

# pre-processed data produced by interval/labeling pipeline
data = pd.read_csv("0_2 merged_respiratory_modelling_final.csv")

data.replace("?", np.nan, inplace=True)
if "Outcome" in data.columns:
    data["Outcome"] = data["Outcome"].replace({"Yes": 1, "No": 0})
    data["Outcome"] = pd.to_numeric(data["Outcome"], errors="coerce")

if "patientid" in data.columns:
    data = data.drop(columns=["patientid"])

if "sex" in data.columns:
    data = pd.get_dummies(data, columns=["sex"], drop_first=True)

for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data.replace([np.inf, -np.inf], np.nan, inplace=True)

X_df = data.drop(columns=["Outcome"]).copy()
y = data["Outcome"].values
feature_names = X_df.columns.to_numpy()
X = X_df.values


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def normalize_weights(weights):
    """Normalize a weight vector to sum to 1."""
    weights = np.clip(weights, 0, 1)
    if weights.sum() == 0:
        return np.ones(len(weights)) / len(weights)
    return weights / weights.sum()


def build_model_spaces():
    """Model definitions and compact grid-search spaces."""
    return {
        "dt": (
            DecisionTreeClassifier(random_state=42),
            {"max_depth": [2, 4, 6, 8, 10]},
        ),
        "rf": (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {"n_estimators": [50, 100, 150], "max_depth": [4, 6, 8, None]},
        ),
        "gb": (
            GradientBoostingClassifier(random_state=42),
            {"learning_rate": [0.01, 0.1, 0.2], "n_estimators": [30, 50]},
        ),
        "knn": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 4, 5]},
        ),
        "lgb": (
            LGBMClassifier(random_state=42, verbose=-1),
            {"learning_rate": [0.01, 0.1, 0.2], "n_estimators": [30, 50]},
        ),
        "xgb": (
            XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            ),
            {"max_depth": [3, 4, 5], "n_estimators": [30, 50]},
        ),
    }


def get_tuned_models(X_train, y_train):
    """Tune each base learner by grid search on training data only."""
    tuned_models = {}
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for name, (model, param_grid) in build_model_spaces().items():
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="f1",
            cv=inner_cv,
            n_jobs=1,
        )
        search.fit(X_train, y_train)
        tuned_models[name] = search.best_estimator_

    return tuned_models


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred: Negative", "Pred: Positive"],
        yticklabels=["True: Negative", "True: Positive"],
    )
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(title)
    plt.show()


def calc_advanced_metrics(y_true, proba, thresh):
    pred = (proba >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    return [thresh, f1_score(y_true, pred), sens, spec, ppv, npv]


# ==========================================
# 3. CROSS-VALIDATION & GENETIC ALGORITHM
# ==========================================
threshold = 0.5
cv_splits = 5
skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

population_size = 20
generations = 30
mutation_rate = 0.1
crossover_rate = 0.8
tournament_size = 4
elitism_count = 5

cv_y_true = []
cv_ga_proba = []
cv_lr_proba = []
cv_rf_proba = []
f1_evolution_all = []
roc_evolution_all = []
combined_feature_importance = np.zeros(len(feature_names))

print("Starting 5-Fold Cross Validation with GA Optimization...")

for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
    print(f"\n--- Fold {fold} ---")
    X_train, X_test = X[train_index], X[val_index]
    y_train, y_test = y[train_index], y[val_index]

    imputer = KNNImputer(n_neighbors=5)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    # Baseline logistic regression (trained only on outer-train)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_smote, y_train_smote)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
    cv_lr_proba.extend(lr_proba)

    tuned_models = get_tuned_models(X_train_smote, y_train_smote)

    # Random Forest baseline on outer-test
    rf_model = clone(tuned_models["rf"])
    rf_model.fit(X_train_smote, y_train_smote)
    rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    cv_rf_proba.extend(rf_proba)

    model_order = ["dt", "rf", "gb", "knn", "lgb", "xgb"]
    test_pred_proba = []
    ensemble_models_full = []

    # Build robust, leakage-safe GA fitness cache via inner CV on outer-train.
    ga_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    inner_cache = []
    base_f1_agg = np.zeros(len(model_order), dtype=float)
    fold_counter = 0

    for ga_train_idx, ga_val_idx in ga_cv.split(X_train_smote, y_train_smote):
        X_ga_train, X_ga_val = X_train_smote[ga_train_idx], X_train_smote[ga_val_idx]
        y_ga_train, y_ga_val = y_train_smote[ga_train_idx], y_train_smote[ga_val_idx]

        split_pred_proba = []
        for m_i, name in enumerate(model_order):
            model_for_ga = clone(tuned_models[name])
            model_for_ga.fit(X_ga_train, y_ga_train)
            pred_inner = model_for_ga.predict_proba(X_ga_val)[:, 1]
            split_pred_proba.append(pred_inner)
            base_f1_agg[m_i] += f1_score(y_ga_val, (pred_inner >= threshold).astype(int))

        inner_cache.append((np.array(split_pred_proba), y_ga_val))
        fold_counter += 1

    base_f1_scores = base_f1_agg / max(fold_counter, 1)

    # Refit tuned models on full outer-train for final outer-test prediction
    for name in model_order:
        model_full = clone(tuned_models[name])
        model_full.fit(X_train_smote, y_train_smote)
        ensemble_models_full.append((name, model_full))
        pred_test = model_full.predict_proba(X_test_scaled)[:, 1]
        test_pred_proba.append(pred_test)
    test_pred_proba = np.array(test_pred_proba)

    def evaluate_fitness_cv(weights):
        w = normalize_weights(weights)
        split_scores = []
        for split_pred_proba, y_inner in inner_cache:
            combined_proba_inner = np.dot(w, split_pred_proba)
            pred_inner = (combined_proba_inner >= threshold).astype(int)
            split_f1 = f1_score(y_inner, pred_inner)
            split_auc = roc_auc_score(y_inner, combined_proba_inner)
            split_scores.append(0.5 * split_f1 + 0.5 * split_auc)
        return float(np.mean(split_scores))

    def local_search_cv(individual, max_iterations=50):
        best_ind = individual.copy()
        best_fit = evaluate_fitness_cv(best_ind)
        for _ in range(max_iterations):
            perturbed = best_ind.copy()
            perturbed[np.random.randint(len(perturbed))] += np.random.normal(0, 0.05)
            perturbed = normalize_weights(perturbed)
            p_fit = evaluate_fitness_cv(perturbed)
            if p_fit > best_fit:
                best_ind, best_fit = perturbed, p_fit
        return best_ind

    population = [normalize_weights(np.array(base_f1_scores))]
    population.extend(np.random.dirichlet(np.ones(len(model_order)), size=population_size - 1))
    population = np.array(population)
    best_weights_fold = None
    best_fitness_fold = -1
    fold_f1_evo = []
    fold_roc_evo = []

    def tournament_select(population_array, fitness_array, k):
        idx = np.random.choice(len(population_array), size=k, replace=False)
        winner = idx[np.argmax(fitness_array[idx])]
        return population_array[winner].copy()

    for generation in range(generations):
        fitness_scores = np.array([evaluate_fitness_cv(ind) for ind in population])
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = population[sorted_indices]
        fitness_scores = fitness_scores[sorted_indices]

        if fitness_scores[0] > best_fitness_fold:
            best_fitness_fold = fitness_scores[0]
            best_weights_fold = population[0].copy()

        # Track inner-CV mean F1/AUC for evolution curves.
        f1_per_split = []
        auc_per_split = []
        for split_pred_proba, y_inner in inner_cache:
            best_proba_gen_inner = np.dot(best_weights_fold, split_pred_proba)
            f1_per_split.append(f1_score(y_inner, (best_proba_gen_inner >= threshold).astype(int)))
            auc_per_split.append(roc_auc_score(y_inner, best_proba_gen_inner))
        fold_f1_evo.append(float(np.mean(f1_per_split)))
        fold_roc_evo.append(float(np.mean(auc_per_split)))

        elite_count = min(elitism_count, population_size)
        for i in range(elite_count):
            population[i] = local_search_cv(population[i])

        next_pop = [population[i].copy() for i in range(elite_count)]
        while len(next_pop) < population_size:
            p1 = tournament_select(population, fitness_scores, tournament_size)
            p2 = tournament_select(population, fitness_scores, tournament_size)
            if np.random.rand() < crossover_rate:
                alpha = np.random.rand()
                c1 = alpha * p1 + (1 - alpha) * p2
                c2 = alpha * p2 + (1 - alpha) * p1
            else:
                c1, c2 = p1.copy(), p2.copy()
            next_pop.extend([normalize_weights(c1), normalize_weights(c2)])

        # Light mutation decay to balance exploration/exploitation.
        current_mutation = mutation_rate * (1.0 - (generation / max(generations - 1, 1)) * 0.5)
        for idx in range(elite_count, len(next_pop)):
            ind = next_pop[idx]
            if np.random.rand() < current_mutation:
                ind[np.random.randint(len(ind))] += np.random.normal(0, 0.1)
                next_pop[idx] = normalize_weights(ind)

        population = np.array(next_pop[:population_size])

    f1_evolution_all.append(fold_f1_evo)
    roc_evolution_all.append(fold_roc_evo)

    for i, (_, model) in enumerate(ensemble_models_full):
        if hasattr(model, "feature_importances_"):
            combined_feature_importance += best_weights_fold[i] * model.feature_importances_

    # Outer-test is used only once, after weights are fixed on inner validation.
    cv_ga_proba.extend(np.dot(best_weights_fold, test_pred_proba))
    cv_y_true.extend(y_test)

cv_y_true = np.array(cv_y_true)
cv_ga_proba = np.array(cv_ga_proba)
cv_lr_proba = np.array(cv_lr_proba)
cv_rf_proba = np.array(cv_rf_proba)


# ==========================================
# 4. RESULTS AND COMPARISONS
# ==========================================
print("\n==========================================")
print("FINAL CROSS-VALIDATION RESULTS")
print("==========================================")

cutoffs = [0.3, 0.5, 0.7]
ga_cutoff_data = [calc_advanced_metrics(cv_y_true, cv_ga_proba, c) for c in cutoffs]
ga_cutoff_df = pd.DataFrame(
    ga_cutoff_data,
    columns=["Threshold", "F1-Score", "Sensitivity", "Specificity", "PPV", "NPV"],
)

print("\n1. GA Ensemble Performance at Clinical Cutoffs:")
print(ga_cutoff_df.to_string(index=False))

baseline_data = [
    ["GA Ensemble"] + calc_advanced_metrics(cv_y_true, cv_ga_proba, 0.5)[1:],
    ["Random Forest"] + calc_advanced_metrics(cv_y_true, cv_rf_proba, 0.5)[1:],
    ["Logistic Regression"] + calc_advanced_metrics(cv_y_true, cv_lr_proba, 0.5)[1:],
]
baseline_df = pd.DataFrame(
    baseline_data,
    columns=["Model", "F1-Score", "Sensitivity", "Specificity", "PPV", "NPV"],
)

print("\n2. Model Comparison (Threshold = 0.5):")
print(baseline_df.to_string(index=False))

all_models_data = []
for c in cutoffs:
    all_models_data.append(["GA Ensemble"] + calc_advanced_metrics(cv_y_true, cv_ga_proba, c))
    all_models_data.append(["Random Forest"] + calc_advanced_metrics(cv_y_true, cv_rf_proba, c))
    all_models_data.append(["Logistic Regression"] + calc_advanced_metrics(cv_y_true, cv_lr_proba, c))

multi_model_df = pd.DataFrame(
    all_models_data,
    columns=["Model", "Threshold", "F1-Score", "Sensitivity", "Specificity", "PPV", "NPV"],
)

print("\n3. Model Comparison Across Multiple Thresholds:")
print(multi_model_df.to_string(index=False))


# ==========================================
# 5. VISUALIZATIONS
# ==========================================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(np.mean(f1_evolution_all, axis=0), label="Mean Max F1 Score (Inner)", color="purple", marker="o")
plt.title("GA F1 Score Evolution (Avg over Folds)")
plt.xlabel("Generation")
plt.ylabel("F1 Score")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(np.mean(roc_evolution_all, axis=0), label="Mean Max ROC AUC (Inner)", color="teal", marker="o")
plt.title("GA ROC AUC Evolution (Avg over Folds)")
plt.xlabel("Generation")
plt.ylabel("ROC AUC")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

fpr_ga, tpr_ga, _ = roc_curve(cv_y_true, cv_ga_proba)
fpr_rf, tpr_rf, _ = roc_curve(cv_y_true, cv_rf_proba)
fpr_lr, tpr_lr, _ = roc_curve(cv_y_true, cv_lr_proba)
axes[0].plot(fpr_ga, tpr_ga, label=f"GA Ensemble (AUC={auc(fpr_ga, tpr_ga):.3f})", color="#1f77b4", lw=2)
axes[0].plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc(fpr_rf, tpr_rf):.3f})", color="#ff7f0e", lw=2)
axes[0].plot(fpr_lr, tpr_lr, label=f"Logistic Reg (AUC={auc(fpr_lr, tpr_lr):.3f})", color="#2ca02c", lw=2)
axes[0].plot([0, 1], [0, 1], "k--")
axes[0].set_title("Cross-Validated ROC Curve")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

ga_frac, ga_mean = calibration_curve(cv_y_true, cv_ga_proba, n_bins=8)
rf_frac, rf_mean = calibration_curve(cv_y_true, cv_rf_proba, n_bins=8)
lr_frac, lr_mean = calibration_curve(cv_y_true, cv_lr_proba, n_bins=8)
axes[1].plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
axes[1].plot(ga_mean, ga_frac, "s-", label="GA Ensemble", color="#1f77b4")
axes[1].plot(rf_mean, rf_frac, "^-", label="Random Forest", color="#ff7f0e")
axes[1].plot(lr_mean, lr_frac, "o-", label="Logistic Reg", color="#2ca02c")
axes[1].set_title("Reliability Diagram (Calibration)")
axes[1].set_xlabel("Mean Predicted Probability")
axes[1].set_ylabel("Actual Fraction of Positives")
axes[1].legend()

min_imp = np.min(combined_feature_importance)
max_imp = np.max(combined_feature_importance)
norm_imp = (combined_feature_importance - min_imp) / (max_imp - min_imp + 1e-9)
top_k = min(15, len(norm_imp))
top_idx = np.argsort(norm_imp)[-top_k:]
axes[2].barh(range(top_k), norm_imp[top_idx], color="teal", align="center")
axes[2].set_yticks(range(top_k))
axes[2].set_yticklabels(feature_names[top_idx])
axes[2].set_title("Normalized Combined Feature Importance")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
plt.show()

plot_confusion_matrix(
    cv_y_true,
    (cv_ga_proba >= 0.5).astype(int),
    title="Aggregated Confusion Matrix (Threshold=0.5)",
)

ga_melted = ga_cutoff_df.melt(id_vars="Threshold", var_name="Metric", value_name="Score")
plt.figure(figsize=(12, 6))
ax1 = sns.barplot(data=ga_melted, x="Metric", y="Score", hue="Threshold", palette="Set2")
plt.title("GA Ensemble Performance Metrics Across Probability Thresholds", fontweight="bold", fontsize=14)
plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
plt.xlabel("Evaluation Metric", fontsize=12)
plt.ylim(0, 1.1)
plt.legend(title="Threshold", loc="upper right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
for container in ax1.containers:
    ax1.bar_label(container, fmt="%.3f", padding=3, fontsize=9, fontweight="bold")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.barplot(data=multi_model_df, x="Threshold", y="F1-Score", hue="Model", ax=axes[0], palette="Set2")
axes[0].set_title("F1-Score Comparison Across Models", fontweight="bold", fontsize=14)
axes[0].set_ylabel("F1-Score", fontsize=12)
axes[0].set_ylim(0, 1.1)
axes[0].grid(axis="y", linestyle="--", alpha=0.7)
for container in axes[0].containers:
    axes[0].bar_label(container, fmt="%.3f", padding=3, fontsize=9, fontweight="bold")

sns.barplot(data=multi_model_df, x="Threshold", y="Sensitivity", hue="Model", ax=axes[1], palette="Set2")
axes[1].set_title("Sensitivity (Recall) Comparison Across Models", fontweight="bold", fontsize=14)
axes[1].set_ylabel("Sensitivity", fontsize=12)
axes[1].set_ylim(0, 1.1)
axes[1].grid(axis="y", linestyle="--", alpha=0.7)
for container in axes[1].containers:
    axes[1].bar_label(container, fmt="%.3f", padding=3, fontsize=9, fontweight="bold")

plt.tight_layout()
plt.show()
