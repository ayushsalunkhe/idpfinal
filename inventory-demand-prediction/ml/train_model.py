"""
train_model.py
----------------
The full training pipeline:
    1. Load raw dataset
    2. Clean it (ml/preprocessing.py)
    3. Engineer features (ml/feature_engineering.py)
    4. Time-based train/test split (NOT random -- avoids leakage)
    5. Train multiple regression models
    6. Evaluate every model with MAE/MSE/RMSE/R2/MAPE
    7. Pick the best model (lowest RMSE on test set)
    8. Save the trained model + scaler + label encoders + metrics
    9. Generate evaluation graphs

Run with:  python ml/train_model.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for server / script use
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from ml.preprocessing import clean_pipeline
from ml.feature_engineering import build_features, FEATURE_COLUMNS, TARGET_COLUMN

DATA_PATH = "data/inventory_data.csv"
MODELS_DIR = "models"
IMAGES_DIR = "static/images"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

sns.set_style("whitegrid")
PALETTE = ["#2563eb", "#16a34a", "#dc2626", "#f59e0b", "#7c3aed", "#0891b2"]


def mean_absolute_percentage_error(y_true, y_pred):
    """MAPE -- expressed as a percentage. Guards against divide-by-zero
    (common with intermittent demand items) by only using rows where
    the true value is non-zero."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def time_based_split(df: pd.DataFrame, test_size: float = 0.2):
    """
    Time-based split instead of random split.
    Why: this is a time-series-like problem. A random split would leak
    "future" information into the training set (e.g. training on data
    from March 2025 while testing on January 2025), giving an unrealistically
    good but misleading test score. We sort by date and take the LAST
    `test_size` fraction of the timeline (per product) as the test set.
    """
    df = df.sort_values("date").reset_index(drop=True)
    cutoff_index = int(len(df) * (1 - test_size))
    cutoff_date = df.iloc[cutoff_index]["date"]

    train_df = df[df["date"] < cutoff_date].copy()
    test_df = df[df["date"] >= cutoff_date].copy()
    return train_df, test_df, cutoff_date


def main():
    print("=" * 70)
    print("STEP 1: Loading & cleaning data")
    print("=" * 70)
    df, encoders = clean_pipeline(DATA_PATH)
    print(f"Cleaned dataset shape: {df.shape}")

    print("\n" + "=" * 70)
    print("STEP 2: Feature engineering")
    print("=" * 70)
    df = build_features(df)
    print(f"Feature-engineered dataset shape: {df.shape}")
    print(f"Features used ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")

    print("\n" + "=" * 70)
    print("STEP 3: Train/test split (time-based, to prevent leakage)")
    print("=" * 70)
    train_df, test_df, cutoff_date = time_based_split(df, test_size=0.2)
    print(f"Cutoff date: {cutoff_date.date()}")
    print(f"Train rows: {len(train_df)}   Test rows: {len(test_df)}")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    # Scale numerical features for Linear Regression (tree models don't need it,
    # but we scale consistently and let tree models ignore the scaling benefit)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n" + "=" * 70)
    print("STEP 4: Training multiple models")
    print("=" * 70)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=12,
                                                random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                                        learning_rate=0.08, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.08,
                                          random_state=42, n_jobs=-1)
    else:
        print("[info] xgboost not installed -- skipping XGBoost (project still works fully without it).")

    results = []
    predictions = {}
    trained_models = {}

    for name, model in models.items():
        print(f"Training {name} ...")
        if name == "Linear Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        preds = np.clip(preds, 0, None)  # demand cannot be negative

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        mape = mean_absolute_percentage_error(y_test, preds)

        results.append({
            "Model": name, "MAE": round(mae, 3), "MSE": round(mse, 3),
            "RMSE": round(rmse, 3), "R2_Score": round(r2, 4), "MAPE_%": round(mape, 2)
        })
        predictions[name] = preds
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    print("\nModel Comparison:")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    print(f"\n>>> Best model selected: {best_model_name} (lowest RMSE) <<<")

    print("\n" + "=" * 70)
    print("STEP 5: Saving model artifacts")
    print("=" * 70)
    joblib.dump(best_model, os.path.join(MODELS_DIR, "demand_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.pkl"))
    joblib.dump(FEATURE_COLUMNS, os.path.join(MODELS_DIR, "feature_columns.pkl"))

    meta = {
        "best_model_name": best_model_name,
        "uses_scaled_input": best_model_name == "Linear Regression",
        "trained_rows": len(train_df),
        "tested_rows": len(test_df),
        "cutoff_date": str(cutoff_date.date()),
    }
    with open(os.path.join(MODELS_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    results_df.to_csv(os.path.join(MODELS_DIR, "model_comparison.csv"), index=False)
    print("Saved: demand_model.pkl, scaler.pkl, encoders.pkl, feature_columns.pkl, model_meta.json")

    print("\n" + "=" * 70)
    print("STEP 6: Generating evaluation graphs")
    print("=" * 70)
    generate_graphs(results_df, y_test, predictions, best_model_name, best_model, train_df, test_df, df)

    print("\nTraining pipeline complete.")
    return results_df, best_model_name


def generate_graphs(results_df, y_test, predictions, best_model_name, best_model,
                     train_df, test_df, full_df):
    # 1. Model performance comparison (RMSE + R2)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(results_df["Model"], results_df["RMSE"], color=PALETTE)
    axes[0].set_title("Model Comparison - RMSE (lower is better)")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(results_df["Model"], results_df["R2_Score"], color=PALETTE)
    axes[1].set_title("Model Comparison - R² Score (higher is better)")
    axes[1].set_ylabel("R² Score")
    axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "model_comparison.png"), dpi=110)
    plt.close()

    # 2. Actual vs Predicted (best model), sample of points for clarity
    best_preds = predictions[best_model_name]
    sample_n = min(300, len(y_test))
    idx = np.random.choice(len(y_test), sample_n, replace=False)
    plt.figure(figsize=(7, 6))
    plt.scatter(np.array(y_test)[idx], best_preds[idx], alpha=0.5, color="#2563eb")
    max_val = max(np.array(y_test)[idx].max(), best_preds[idx].max())
    plt.plot([0, max_val], [0, max_val], "r--", label="Perfect Prediction")
    plt.xlabel("Actual Demand")
    plt.ylabel("Predicted Demand")
    plt.title(f"Actual vs Predicted Demand ({best_model_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "actual_vs_predicted.png"), dpi=110)
    plt.close()

    # 3. Feature importance (only for tree-based models)
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=FEATURE_COLUMNS)
        importances = importances.sort_values(ascending=True).tail(15)
        plt.figure(figsize=(8, 7))
        importances.plot(kind="barh", color="#16a34a")
        plt.title(f"Feature Importance ({best_model_name})")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, "feature_importance.png"), dpi=110)
        plt.close()

    # 4. Demand trend over time (overall, monthly aggregated)
    monthly = full_df.copy()
    monthly["year_month"] = monthly["date"].dt.to_period("M").astype(str)
    monthly_trend = monthly.groupby("year_month")["units_sold"].sum().reset_index()
    plt.figure(figsize=(11, 5))
    plt.plot(monthly_trend["year_month"], monthly_trend["units_sold"], marker="o", color="#7c3aed")
    plt.xticks(rotation=60)
    plt.title("Overall Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Total Units Sold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "monthly_sales_trend.png"), dpi=110)
    plt.close()

    # 5. Product-wise demand (top 10 products by total units sold)
    product_totals = full_df.groupby("product_name")["units_sold"].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(9, 6))
    product_totals.sort_values().plot(kind="barh", color="#f59e0b")
    plt.title("Top 10 Products by Total Units Sold")
    plt.xlabel("Total Units Sold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "product_wise_demand.png"), dpi=110)
    plt.close()

    print("Saved graphs to static/images/: model_comparison.png, actual_vs_predicted.png, "
          "feature_importance.png, monthly_sales_trend.png, product_wise_demand.png")


if __name__ == "__main__":
    main()
