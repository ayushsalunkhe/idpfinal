"""
evaluate_model.py
-------------------
Lightweight helpers for loading saved evaluation results (produced during
training) so the Flask app can display them without re-running training.
"""

import os
import json
import pandas as pd

MODELS_DIR = "models"


def load_model_comparison() -> pd.DataFrame:
    """Load the model comparison table saved during training."""
    path = os.path.join(MODELS_DIR, "model_comparison.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def load_model_meta() -> dict:
    """Load metadata about which model was selected as best, and why."""
    path = os.path.join(MODELS_DIR, "model_meta.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def explain_best_model() -> str:
    """
    Human-readable explanation of why the selected model performs best.
    Used in the README / analytics page.
    """
    meta = load_model_meta()
    comparison = load_model_comparison()
    if meta.get("best_model_name") is None or comparison.empty:
        return "Model has not been trained yet. Run `python ml/train_model.py` first."

    best = meta["best_model_name"]
    row = comparison[comparison["Model"] == best].iloc[0]

    explanation = (
        f"'{best}' was selected as the best model because it achieved the lowest "
        f"RMSE ({row['RMSE']}) and the highest R2 score ({row['R2_Score']}) on the "
        f"held-out (time-based) test set among all models compared. "
    )
    if best in ("Random Forest", "Gradient Boosting", "XGBoost", "Decision Tree"):
        explanation += (
            "Tree-based ensemble models like this tend to outperform Linear Regression "
            "on retail demand data because demand depends on NON-LINEAR interactions "
            "between features (e.g. a promotion during a festive weekend has a much "
            "bigger combined effect than the sum of its individual effects), which "
            "linear models cannot capture but tree-based models can."
        )
    else:
        explanation += (
            "In this run, the relationship between the engineered features and demand "
            "was captured well enough by a linear model, though tree-based models are "
            "usually preferred for retail demand forecasting in production."
        )
    return explanation
