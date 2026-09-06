"""
prediction.py
--------------
Uses the TRAINED model (saved by train_model.py) to:
    1. Predict future demand for a product
    2. Classify stockout risk (LOW / MEDIUM / HIGH)
    3. Recommend a reorder quantity and status

-------------------------------------------------------------------------
Demand Prediction vs Stockout Prediction -- what's the difference?
-------------------------------------------------------------------------
- DEMAND PREDICTION is a regression problem: "How many units of this
  product will customers buy tomorrow?" The trained ML model (Random
  Forest / Gradient Boosting / etc.) outputs a number, e.g. 75 units.

- STOCKOUT PREDICTION is a decision/classification problem built ON TOP
  of the demand prediction: "Given that predicted demand, the current
  stock on hand, and how long it takes the supplier to deliver more
  stock (lead time), will this product run out before the next
  delivery arrives?" It outputs a category: LOW / MEDIUM / HIGH risk.

  We compute stockout risk with clear, auditable business logic (not a
  black box), using the ML-predicted demand as its main input. This is
  the standard approach used in real inventory systems, because
  reorder decisions need to be explainable to a store manager.
-------------------------------------------------------------------------
"""

import os
import joblib
import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(__file__)
# models directory is a sibling of the `ml` package directory
MODELS_DIR = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, "models"))

_model = None
_scaler = None
_encoders = None
_feature_columns = None
_meta = None


def _load_artifacts():
    """Lazy-load model artifacts once and cache them."""
    global _model, _scaler, _encoders, _feature_columns, _meta
    if _model is None:
        _model = joblib.load(os.path.join(MODELS_DIR, "demand_model.pkl"))
        _scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        _encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
        _feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
        import json
        with open(os.path.join(MODELS_DIR, "model_meta.json")) as f:
            _meta = json.load(f)
    return _model, _scaler, _encoders, _feature_columns, _meta


def predict_demand_from_features(feature_row: dict) -> float:
    """
    Predict future demand given a dict of feature values.
    feature_row keys must match ml.feature_engineering.FEATURE_COLUMNS.
    """
    model, scaler, encoders, feature_columns, meta = _load_artifacts()

    X = pd.DataFrame([feature_row])[feature_columns]

    if meta.get("uses_scaled_input"):
        X = scaler.transform(X)
        pred = model.predict(X)[0]
    else:
        pred = model.predict(X)[0]

    return max(0.0, round(float(pred), 2))


def build_feature_row_for_product(df: pd.DataFrame, product_id: int,
                                   as_of_date: pd.Timestamp = None) -> dict:
    """
    Build the latest available feature row for a product from historical
    data, so we can predict "tomorrow's" demand using data known up to
    `as_of_date` (defaults to the most recent date on record).

    df must already be the FEATURE-ENGINEERED dataframe (see
    ml.feature_engineering.build_features), i.e. it has category_encoded,
    rolling_7day_sales, etc. already computed.
    """
    product_df = df[df["product_id"] == product_id].sort_values("date")
    if product_df.empty:
        raise ValueError(f"No historical data found for product_id={product_id}")

    if as_of_date is not None:
        product_df = product_df[product_df["date"] <= as_of_date]
        if product_df.empty:
            raise ValueError(f"No data for product_id={product_id} on/before {as_of_date}")

    latest = product_df.iloc[-1].to_dict()
    return latest


def calculate_safety_stock(daily_demand_std: float, lead_time: float, service_z: float = 1.65) -> float:
    """Safety stock for daily demand over the lead-time window."""
    if daily_demand_std <= 0:
        daily_demand_std = 0.1
    return round(service_z * daily_demand_std * np.sqrt(max(lead_time, 1)), 2)


def calculate_reorder_point(predicted_monthly_demand: float, lead_time_days: float, safety_stock: float) -> float:
    """Convert monthly forecast to daily demand before lead-time math."""
    daily_demand = max(predicted_monthly_demand / 30.0, 0.0)
    expected_demand_during_lead_time = daily_demand * lead_time_days
    return round(expected_demand_during_lead_time + safety_stock, 2)


def classify_stockout_risk(current_stock: float, reorder_point: float) -> str:
    """Risk based on stock relative to reorder point. This prevents making every product high risk."""
    if reorder_point <= 0:
        return "LOW RISK"

    stock_ratio = current_stock / reorder_point
    if stock_ratio < 0.5:
        return "HIGH RISK"
    elif stock_ratio < 1.0:
        return "MEDIUM RISK"
    return "LOW RISK"


def recommend_reorder(current_stock: float, predicted_daily_demand: float,
                       lead_time: float, daily_demand_std: float) -> dict:
    """Reorder recommendation using consistent units and stock-to-reorder-point logic."""
    safety_stock = calculate_safety_stock(daily_demand_std, lead_time)
    reorder_point = calculate_reorder_point(predicted_daily_demand, lead_time, safety_stock)
    risk = classify_stockout_risk(current_stock, reorder_point)

    recommended_qty = max(0, round(reorder_point - current_stock, 2))
    status = "REORDER REQUIRED" if current_stock <= reorder_point else "STOCK SUFFICIENT"

    return {
        "predicted_demand": predicted_daily_demand,
        "current_stock": current_stock,
        "supplier_lead_time": lead_time,
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "recommended_reorder_qty": recommended_qty,
        "stockout_risk": risk,
        "status": status,
    }


def predict_for_product(df: pd.DataFrame, product_id: int, as_of_date: pd.Timestamp = None) -> dict:
    """
    End-to-end convenience function: given the feature-engineered
    dataframe and a product_id, returns the complete prediction +
    reorder recommendation, ready to display in the web app.
    """
    feature_row = build_feature_row_for_product(df, product_id, as_of_date)
    _, _, _, feature_columns, _ = _load_artifacts()

    input_features = {col: feature_row[col] for col in feature_columns}
    predicted_demand = predict_demand_from_features(input_features)

    product_history = df[df["product_id"] == product_id]
    daily_demand_std = float(product_history["units_sold"].std(skipna=True) or 0)

    result = recommend_reorder(
        current_stock=float(feature_row["current_stock"]),
        predicted_daily_demand=predicted_demand,
        lead_time=float(feature_row["supplier_lead_time"]),
        daily_demand_std=daily_demand_std,
    )
    result["product_id"] = product_id
    result["product_name"] = feature_row.get("product_name", "")
    result["category"] = feature_row.get("category", "")
    result["as_of_date"] = str(feature_row["date"].date()) if hasattr(feature_row["date"], "date") else str(feature_row["date"])
    return result
