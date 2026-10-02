#!/usr/bin/env python3
"""
model/train_model.py
====================
Complete ML training pipeline for the Crop Recommendation System.

Steps
-----
1. Load dataset
2. Exploratory Data Analysis
3. Preprocessing
4. Model training (Logistic Regression, Decision Tree, Random Forest, KNN)
5. Evaluation and model selection
6. Save final model and scaler

Usage
-----
    python model/train_model.py

Dataset
-------
Place the dataset at:  data/Crop_recommendation.csv

Source:  https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
"""

import os
import sys
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
matplotlib.use("Agg")  # Non-interactive backend -- safe for server environments

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(BASE_DIR, "data", "Crop_recommendation.csv")
MODEL_DIR    = os.path.join(BASE_DIR, "model")
FIGURES_DIR  = os.path.join(BASE_DIR, "reports", "figures")
MODEL_PATH   = os.path.join(MODEL_DIR, "crop_model.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COL   = "label"
RANDOM_STATE = 42
TEST_SIZE    = 0.2


# ── Helpers ──────────────────────────────────────────────────────

def ensure_dirs():
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def save_figure(fig, name: str):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


# ── Step 1 -- Load Dataset ────────────────────────────────────────

def load_data() -> pd.DataFrame:
    print("\n" + "="*60)
    print("STEP 1 -- LOAD DATASET")
    print("="*60)

    if not os.path.exists(DATA_PATH):
        print(f"\n[ERROR] Dataset not found at: {DATA_PATH}")
        print("\nPlease download the dataset from:")
        print("  https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
        print(f"\nAnd place it at: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"\nDataset loaded successfully.")
    print(f"  Rows     : {df.shape[0]:,}")
    print(f"  Columns  : {df.shape[1]}")
    print(f"  Features : {FEATURE_COLS}")
    print(f"  Target   : {TARGET_COL}")
    print(f"  Crops    : {sorted(df[TARGET_COL].unique().tolist())}")
    print(f"  Classes  : {df[TARGET_COL].nunique()}")
    return df


# ── Step 2 -- EDA ─────────────────────────────────────────────────

def perform_eda(df: pd.DataFrame):
    print("\n" + "="*60)
    print("STEP 2 -- EXPLORATORY DATA ANALYSIS")
    print("="*60)

    print("\n--- Shape ---")
    print(f"  {df.shape}")

    print("\n--- Missing values ---")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "  None found.")

    print("\n--- Duplicates ---")
    dups = df.duplicated().sum()
    print(f"  {dups} duplicate rows.")

    print("\n--- Statistical Summary ---")
    print(df.describe().to_string())

    print("\n--- Class Distribution ---")
    print(df[TARGET_COL].value_counts().to_string())

    # ── Figure 1: Crop Distribution ──────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    counts = df[TARGET_COL].value_counts()
    ax.bar(counts.index, counts.values, color=sns.color_palette("Greens_d", len(counts)))
    ax.set_title("Crop Distribution in Dataset", fontsize=14, fontweight="bold")
    ax.set_xlabel("Crop")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_figure(fig, "01_crop_distribution.png")

    # ── Figure 2: Correlation Heatmap ────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df[FEATURE_COLS].corr()
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="YlGn", ax=ax,
        linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, "02_correlation_heatmap.png")

    # ── Figure 3: Feature Distributions ──────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        axes[i].hist(df[col], bins=30, color="#4CAF50", edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")
    axes[-1].axis("off")
    fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, "03_feature_distributions.png")

    # ── Figure 4: Boxplots ────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        axes[i].boxplot(df[col], patch_artist=True,
                        boxprops=dict(facecolor="#A5D6A7"))
        axes[i].set_title(col, fontsize=11)
    axes[-1].axis("off")
    fig.suptitle("Feature Boxplots", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, "04_feature_boxplots.png")

    print("\n  EDA figures saved to reports/figures/")


# ── Step 3 -- Preprocessing ───────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """
    Preprocessing steps:
    - Drop duplicates  (prevents data leakage from identical rows)
    - Separate features / target
    - Train/test split  (stratified to maintain class balance)
    - Standard scaling  (required for LogReg and KNN; harmless for tree models)
    """
    print("\n" + "="*60)
    print("STEP 3 -- PREPROCESSING")
    print("="*60)

    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"\n  Duplicates removed : {before - after}")
    print(f"  Remaining rows     : {after:,}")

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Train size : {len(X_train):,}")
    print(f"  Test size  : {len(X_test):,}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Save scaler immediately -- must use exact same object during inference
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n  Scaler saved to {SCALER_PATH}")

    return X_train_sc, X_test_sc, y_train, y_test, scaler


# ── Step 4 -- Train Models ────────────────────────────────────────

def train_models(X_train, y_train) -> dict:
    """Train all four classifiers."""
    print("\n" + "="*60)
    print("STEP 4 -- MODEL TRAINING")
    print("="*60)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    trained = {}
    for name, clf in models.items():
        print(f"\n  Training {name} ...", end=" ", flush=True)
        clf.fit(X_train, y_train)
        trained[name] = clf
        print("done.")

    return trained


# ── Step 5 -- Evaluate Models ─────────────────────────────────────

def evaluate_models(trained: dict, X_test, y_test) -> str:
    """
    Evaluate all models and print a comparison table.
    Returns the name of the best model based on weighted F1-score.

    Weighted averaging is chosen because:
    - The dataset may have slight class-size imbalances.
    - Weighted F1 accounts for each class's support, giving a realistic
      overall picture rather than treating all classes equally (macro).
    """
    print("\n" + "="*60)
    print("STEP 5 -- MODEL EVALUATION")
    print("="*60)

    results = {}
    for name, clf in trained.items():
        y_pred = clf.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        results[name] = dict(accuracy=acc, precision=prec, recall=rec, f1=f1)

    # ── Comparison table ─────────────────────────────────────────
    print(f"\n{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1-score':>10}")
    print("-" * 63)
    for name, m in results.items():
        print(
            f"  {name:<20} {m['accuracy']*100:>8.2f}%"
            f" {m['precision']*100:>9.2f}%"
            f" {m['recall']*100:>7.2f}%"
            f" {m['f1']*100:>9.2f}%"
        )

    # Select best model
    best_name = max(results, key=lambda n: results[n]["f1"])
    print(f"\n  [Best] Best model (weighted F1): {best_name}")

    # ── Figure 5: Metrics bar chart ───────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    names   = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-score"]
    x       = np.arange(len(names))
    width   = 0.2
    colors  = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [results[n][metric] * 100 for n in names]
        ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, rotation=10)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.7)
    plt.tight_layout()
    save_figure(fig, "05_model_comparison.png")

    # ── Figure 6: Best model confusion matrix ─────────────────────
    best_clf = trained[best_name]
    y_pred_best = best_clf.predict(X_test)
    classes = sorted(np.unique(y_test))
    cm = confusion_matrix(y_test, y_pred_best, labels=classes)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=classes, yticklabels=classes,
        ax=ax, linewidths=0.3,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"Confusion Matrix -- {best_name}", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_figure(fig, "06_confusion_matrix.png")

    # ── Figure 7: Feature importance (Random Forest) ──────────────
    if "Random Forest" in trained:
        rf = trained["Random Forest"]
        importances = rf.feature_importances_
        idx = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(
            [FEATURE_COLS[i] for i in idx],
            [importances[i] for i in idx],
            color=sns.color_palette("Greens_d", len(FEATURE_COLS)),
        )
        ax.set_title("Feature Importances -- Random Forest", fontsize=13, fontweight="bold")
        ax.set_ylabel("Importance")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        save_figure(fig, "07_feature_importance.png")

    # Detailed report for best model
    print(f"\n--- Detailed Classification Report: {best_name} ---")
    print(classification_report(y_test, y_pred_best, zero_division=0))

    # Save evaluation summary as text
    _save_eval_summary(results, best_name)

    return best_name


def _save_eval_summary(results: dict, best_name: str):
    """Write model evaluation results to a text file."""
    out_path = os.path.join(FIGURES_DIR, "evaluation_summary.txt")
    lines = ["MODEL EVALUATION SUMMARY", "=" * 50, ""]
    lines.append(
        f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1-score':>10}"
    )
    lines.append("-" * 63)
    for name, m in results.items():
        marker = " << SELECTED" if name == best_name else ""
        lines.append(
            f"  {name:<20} {m['accuracy']*100:>8.2f}%"
            f" {m['precision']*100:>9.2f}%"
            f" {m['recall']*100:>7.2f}%"
            f" {m['f1']*100:>9.2f}%"
            f"{marker}"
        )
    lines += ["", f"Selected model: {best_name}", ""]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  [saved] Evaluation summary -> {out_path}")


# ── Step 6 -- Save Model ──────────────────────────────────────────

def save_model(trained: dict, best_name: str):
    print("\n" + "="*60)
    print("STEP 6 -- SAVE MODEL")
    print("="*60)

    best_clf = trained[best_name]
    joblib.dump(best_clf, MODEL_PATH)
    print(f"\n  Model saved  -> {MODEL_PATH}")
    print(f"  Scaler saved -> {SCALER_PATH}  (saved in Step 3)")
    print(f"\n  Selected model : {best_name}")
    if hasattr(best_clf, "classes_"):
        print(f"  Crop classes   : {len(best_clf.classes_)}")
        print(f"  Classes        : {sorted(best_clf.classes_.tolist())}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    ensure_dirs()
    df = load_data()
    perform_eda(df)
    X_train, X_test, y_train, y_test, scaler = preprocess(df)
    trained = train_models(X_train, y_train)
    best_name = evaluate_models(trained, X_test, y_test)
    save_model(trained, best_name)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nYou can now start the Flask application:")
    print("  python app.py")
    print()


if __name__ == "__main__":
    main()
