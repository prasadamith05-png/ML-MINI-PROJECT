"""
Main Training, Evaluation, and Benchmarking Pipeline
for College Football Bowl Game Predictor.

Trains and validates all Stanford CS229 models + modern extensions,
computes Pick'em pool confidence scores, generates publication figures,
and saves artifacts for interactive demonstration.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure local imports work cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import (
    load_dataset, split_data_temporal, prepare_differential_features, prepare_raw_score_features, FEATURE_COLS
)
from src.models import (
    BaselineModel, ScoreRegressionModel, DifferentialRegressionModel,
    LogisticRegressionClassifier, SupportVectorMachineClassifier,
    RandomForestModel, GradientBoostingModel
)
from src.feature_selection import (
    run_forward_stepwise_selection, run_lasso_sparsity_selection, analyze_feature_correlations
)
from src.evaluate import (
    evaluate_classification, evaluate_regression, simulate_bowl_pickem_pool, compute_roc_curve
)

sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13})


def train_and_evaluate_all():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(base_dir, "figures")
    models_dir = os.path.join(base_dir, "models")
    deliverables_dir = os.path.join(base_dir, "deliverables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(deliverables_dir, exist_ok=True)

    print("=" * 80)
    print("COLLEGE FOOTBALL BOWL GAME PREDICTOR - TRAINING & BENCHMARK SUITE")
    print("=" * 80)

    # 1. Load Data & Time-Aware Split
    df = load_dataset()
    print(f"Total historical bowl games loaded: {len(df)}")
    train_df, val_df, test_df = split_data_temporal(df, train_end_year=2018, val_end_year=2021)
    print(f"Dataset split: Train (2000-2018)={len(train_df)}, Val (2019-2021)={len(val_df)}, Test (2022-2023)={len(test_df)}")

    # 2. Prepare Features
    diff_data = prepare_differential_features(train_df, val_df, test_df, scale=True)
    raw_data = prepare_raw_score_features(train_df, val_df, test_df)

    X_train, y_reg_train, y_clf_train = diff_data["X_train"], diff_data["y_reg_train"], diff_data["y_clf_train"]
    X_test, y_reg_test, y_clf_test = diff_data["X_test"], diff_data["y_reg_test"], diff_data["y_clf_test"]
    scaler = diff_data["scaler"]

    # 3. Initialize Models
    models = {
        "Baseline (Win %)": BaselineModel(),
        "CS229 Score Regression": ScoreRegressionModel(),
        "Diff Linear Reg (OLS)": DifferentialRegressionModel(method="ols"),
        "Diff Ridge Reg": DifferentialRegressionModel(method="ridge", alpha=2.0),
        "Diff Lasso Reg": DifferentialRegressionModel(method="lasso", alpha=0.2),
        "Logistic Regression": LogisticRegressionClassifier(C=1.0),
        "Linear SVM": SupportVectorMachineClassifier(kernel="linear", C=1.0),
        "RBF Kernel SVM": SupportVectorMachineClassifier(kernel="rbf", C=1.0, gamma=0.08),
        "Random Forest": RandomForestModel(n_estimators=60),
        "Gradient Boosting": GradientBoostingModel(n_estimators=50, learning_rate=0.08)
    }

    results = []
    roc_dict = {}
    preds_dict = {}

    # 4. Train & Evaluate Each Model
    for name, model in models.items():
        print(f"Training {name}...")
        if name == "Baseline (Win %)":
            pred_winner, pred_spread = model.predict(test_df)
            win_prob = 1.0 / (1.0 + np.exp(-pred_spread / 10.0))
        elif name == "CS229 Score Regression":
            model.fit(raw_data["X_a_train"], raw_data["y_a_train"], raw_data["X_b_train"], raw_data["y_b_train"])
            pred_winner, pred_spread, win_prob, _, _ = model.predict(raw_data["X_a_test"], raw_data["X_b_test"])
        elif isinstance(model, DifferentialRegressionModel):
            model.fit(X_train, y_reg_train)
            pred_winner, pred_spread, win_prob = model.predict(X_test)
        elif isinstance(model, (LogisticRegressionClassifier, SupportVectorMachineClassifier)):
            model.fit(X_train, y_clf_train)
            pred_winner, pred_spread, win_prob = model.predict(X_test)
        elif isinstance(model, (RandomForestModel, GradientBoostingModel)):
            model.fit(X_train, y_clf_train, y_reg_train)
            pred_winner, pred_spread, win_prob = model.predict(X_test)

        # Metrics
        clf_metrics = evaluate_classification(y_clf_test, pred_winner, win_prob)
        reg_metrics = evaluate_regression(y_reg_test, pred_spread)
        pool_pts, max_pts, pool_pct = simulate_bowl_pickem_pool(y_clf_test, pred_winner, win_prob)

        preds_dict[name] = {"pred_winner": pred_winner, "pred_spread": pred_spread, "win_prob": win_prob}

        fpr, tpr, roc_auc = compute_roc_curve(y_clf_test, win_prob)
        roc_dict[name] = (fpr, tpr, roc_auc)

        results.append({
            "Model": name,
            "Accuracy (%)": clf_metrics["accuracy"],
            "ROC-AUC": clf_metrics["roc_auc"],
            "F1-Score": clf_metrics["f1"],
            "Brier Score": clf_metrics["brier_score"],
            "Spread MAE (pts)": reg_metrics["mae"],
            "Spread RMSE (pts)": reg_metrics["rmse"],
            "Pick'em Pool Pts": f"{pool_pts}/{max_pts} ({pool_pct}%)",
            "Pick'em Pct": pool_pct
        })

    results_df = pd.DataFrame(results).sort_values(by="Accuracy (%)", ascending=False)
    print("\n" + "=" * 80)
    print("EXPERIMENTAL RESULTS TABLE (TEST SET: 2022-2023 BOWL GAMES)")
    print("=" * 80)
    print(results_df.to_string(index=False))

    results_df.to_csv(os.path.join(deliverables_dir, "model_benchmark_results.csv"), index=False)

    # 5. Feature Selection
    print("\n" + "=" * 80)
    print("STANFORD CS229 FEATURE SELECTION ANALYSIS")
    print("=" * 80)
    selected_features, mask = run_forward_stepwise_selection(X_train, y_reg_train, FEATURE_COLS, n_features_to_select=6)
    print(f"Top 6 Features from Forward Stepwise Selection: {selected_features}")

    active_lasso, lasso_alpha = run_lasso_sparsity_selection(X_train, y_reg_train, FEATURE_COLS, alpha=0.20)
    print(f"\nLasso L1 Non-Zero Coefficients (Alpha={lasso_alpha}):")
    print(active_lasso.to_string())

    corrs = analyze_feature_correlations(df, FEATURE_COLS)
    print("\nFeature Pearson Correlation with Point Differential:")
    print(corrs.to_string())

    # 6. Figures
    print("\nGenerating publication figures...")

    # Figure 1: Model Comparison
    fig, ax1 = plt.subplots(figsize=(10, 5.2))
    x_pos = np.arange(len(results_df))
    width = 0.38
    rects1 = ax1.bar(x_pos - width/2, results_df["Accuracy (%)"], width, label="Winner Accuracy (%)", color="#1f77b4")
    rects2 = ax1.bar(x_pos + width/2, results_df["Pick'em Pct"], width, label="Pick'em Pool Score (%)", color="#2ca02c")
    ax1.set_ylabel("Performance (%)")
    ax1.set_title("NCAA Bowl Game Predictor: Model Accuracy & Pick'em Pool Score")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(results_df["Model"], rotation=30, ha="right")
    ax1.set_ylim(40, 78)
    ax1.axhline(50.0, color="gray", linestyle="--", alpha=0.7, label="Coin-Flip Baseline (50%)")
    ax1.axhline(66.0, color="crimson", linestyle=":", linewidth=2, label="Stanford CS229 Benchmark (66%)")
    ax1.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "model_comparison.png"), dpi=300)
    plt.close()

    # Figure 2: Feature Importance
    fig, ax = plt.subplots(figsize=(9, 5.2))
    rf_weights = models["Random Forest"].feature_importances_
    feat_series = pd.Series(rf_weights, index=FEATURE_COLS).sort_values(ascending=True)
    feat_series.plot(kind="barh", ax=ax, color="#2b5c8f")
    ax.set_title("Feature Importance Ranking (Random Forest Impurity Reduction)")
    ax.set_xlabel("Relative Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "feature_importance.png"), dpi=300)
    plt.close()

    # Figure 3: ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for m_name in ["Diff Linear Reg (OLS)", "Diff Ridge Reg", "Logistic Regression", "Linear SVM", "Random Forest"]:
        if m_name in roc_dict:
            fpr, tpr, roc_auc = roc_dict[m_name]
            ax.plot(fpr, tpr, label=f"{m_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "roc_curves.png"), dpi=300)
    plt.close()

    # Figure 4: Point Spread Scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    best_spread = preds_dict["Diff Linear Reg (OLS)"]["pred_spread"]
    ax.scatter(y_reg_test, best_spread, alpha=0.7, color="#1b4f72", edgecolors="white", s=60)
    lims = [-35, 35]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Calibration (y = x)")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual Point Differential (Team A - Team B)")
    ax.set_ylabel("Predicted Point Spread (Team A - Team B)")
    ax.set_title("Differential Linear Regression: Predicted vs Actual Spread")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "spread_prediction_scatter.png"), dpi=300)
    plt.close()

    # 7. Save Pipeline Artifacts
    pipeline_data = {
        "models": models,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "results_df": results_df,
        "best_model_name": "Diff Linear Reg (OLS)"
    }
    with open(os.path.join(models_dir, "trained_bowl_predictor_pipeline.pkl"), "wb") as f:
        pickle.dump(pipeline_data, f)

    print("Pipeline artifacts saved to models/trained_bowl_predictor_pipeline.pkl")
    print("=" * 80)
    print("ALL TRAINING & EVALUATION SUCCESSFULLY FINISHED!")
    print("=" * 80)


if __name__ == "__main__":
    train_and_evaluate_all()
