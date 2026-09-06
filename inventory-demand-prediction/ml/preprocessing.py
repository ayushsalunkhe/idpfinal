"""
preprocessing.py
-----------------
Handles all data cleaning steps BEFORE feature engineering.

Each function documents *why* the step is needed, not just *what* it does,
since this project is meant to demonstrate understanding of the ML pipeline
(important for a college viva / presentation).
"""

import pandas as pd
import numpy as np


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    df = pd.read_csv(path)
    return df


def convert_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the 'date' column from string -> datetime.
    Why: pandas datetime objects let us extract day/month/year/week/quarter
    later and sort the data chronologically, which is essential for a
    time-series style demand prediction problem.
    """
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows.
    Why: duplicate rows (e.g. from double data-entry or a logging bug)
    would bias the model by over-weighting those particular observations
    during training.
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"[preprocessing] Removed {before - after} duplicate rows.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values column by column.
    Why: most ML algorithms (Linear Regression, tree models via sklearn)
    cannot handle NaNs directly. We use sensible statistical imputation
    rather than simply dropping rows, to avoid losing valuable historical
    sales data.
    """
    missing_before = df.isnull().sum().sum()

    # Numerical columns -> median imputation per product (robust to outliers)
    numeric_cols = ["current_stock", "units_sold", "price", "discount",
                     "supplier_lead_time", "supplier_rating"]
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df.groupby("product_id")[col].transform(
                lambda x: x.fillna(x.median())
            )
            # fallback: global median if a product had ALL nulls for that column
            df[col] = df[col].fillna(df[col].median())

    # Categorical columns -> mode imputation
    categorical_cols = ["category", "season"]
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    missing_after = df.isnull().sum().sum()
    print(f"[preprocessing] Missing values: {missing_before} -> {missing_after}")
    return df


def remove_outliers_iqr(df: pd.DataFrame, column: str = "units_sold",
                         group_col: str = "product_id", factor: float = 3.0) -> pd.DataFrame:
    """
    Detect and cap outliers using the IQR (Interquartile Range) method,
    computed PER PRODUCT (since different products have very different
    demand scales -- 2 units/day for earphones vs 20 units/day for rice).

    Why cap instead of drop: dropping rows creates gaps in the daily time
    series, which breaks rolling-window feature engineering (rolling
    average, lag features). Capping keeps the row but limits the extreme
    value's influence on the model.
    """
    q1 = df.groupby(group_col)[column].transform(lambda x: x.quantile(0.25))
    q3 = df.groupby(group_col)[column].transform(lambda x: x.quantile(0.75))
    iqr = q3 - q1
    lower = (q1 - factor * iqr).clip(lower=0)
    upper = q3 + factor * iqr

    n_outliers = int(((df[column] < lower) | (df[column] > upper)).sum())
    df[column] = df[column].clip(lower=lower, upper=upper, axis=0)

    print(f"[preprocessing] Capped {n_outliers} outlier values in '{column}'.")
    return df


def encode_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Encode categorical text columns into numbers using Label Encoding.
    Why: ML models need numeric input. We use label encoding (not one-hot)
    for 'category' and 'season' since tree-based models (Random Forest,
    Gradient Boosting) handle label-encoded categories well and it keeps
    the feature count manageable.

    Returns the dataframe plus a dict of fitted encoders so the SAME
    mapping can be reused at prediction time (very important -- using a
    fresh encoder at prediction time would scramble the category ids).
    """
    from sklearn.preprocessing import LabelEncoder

    encoders = {}
    for col in ["category", "season"]:
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def clean_pipeline(path: str) -> tuple[pd.DataFrame, dict]:
    """Run the full cleaning pipeline in the correct order."""
    df = load_raw_data(path)
    df = convert_date_column(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = remove_outliers_iqr(df, column="units_sold")
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    df, encoders = encode_categorical(df)
    return df, encoders
