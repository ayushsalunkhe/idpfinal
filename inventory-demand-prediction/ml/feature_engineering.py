"""
feature_engineering.py
------------------------
Creates model-ready features from the cleaned dataset, and builds the
target variable `future_demand`.

IMPORTANT - Avoiding data leakage:
    Every feature here is computed using ONLY information available
    STRICTLY BEFORE the row's own date for that product (shift(1) or
    earlier). The target `future_demand` is the NEXT day's units_sold,
    shifted the opposite direction. This mirrors a real deployment
    scenario: on day T, you only know history up to day T, and you are
    trying to predict demand for day T+1.
"""

import pandas as pd
import numpy as np


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Day/Month/Year/Week/Quarter -- capture seasonality & calendar effects."""
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling / lag features computed PER PRODUCT, using only past values
    (shift(1) before rolling) so today's row never uses today's own sales.
    """
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    grouped = df.groupby("product_id")["units_sold"]

    # Previous day / previous week sales (lag features)
    df["previous_day_sales"] = grouped.shift(1)
    df["previous_week_sales"] = grouped.shift(7)

    # Rolling averages using only PAST data: shift(1) first, then roll
    shifted = grouped.shift(1)
    df["rolling_7day_sales"] = shifted.groupby(df["product_id"]).transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    df["rolling_30day_sales"] = shifted.groupby(df["product_id"]).transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )

    # Average historical demand up to (but not including) this row
    df["avg_historical_demand"] = shifted.groupby(df["product_id"]).transform(
        lambda x: x.expanding(min_periods=1).mean()
    )

    # Sales growth rate: (recent 7-day avg - previous 7-day avg) / previous 7-day avg
    prev_7 = df["rolling_7day_sales"]
    prev_7_shifted = df.groupby("product_id")["rolling_7day_sales"].shift(7)
    df["sales_growth_rate"] = ((prev_7 - prev_7_shifted) / prev_7_shifted.replace(0, np.nan))
    df["sales_growth_rate"] = df["sales_growth_rate"].fillna(0)

    # Stock-to-demand ratio: current stock relative to recent average demand
    # (uses current_stock, which is known at the START of the day, and
    #  historical demand only -- no leakage of today's sales)
    df["stock_to_demand_ratio"] = df["current_stock"] / df["avg_historical_demand"].replace(0, np.nan)
    df["stock_to_demand_ratio"] = df["stock_to_demand_ratio"].fillna(df["current_stock"])

    # Fill any remaining NaNs (start-of-series rows with no history yet)
    fill_cols = ["previous_day_sales", "previous_week_sales", "rolling_7day_sales",
                 "rolling_30day_sales", "avg_historical_demand"]
    for col in fill_cols:
        df[col] = df.groupby("product_id")[col].transform(lambda x: x.fillna(x.mean()))
        df[col] = df[col].fillna(0)

    return df


def add_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    future_demand = next day's units_sold for the SAME product.
    This is the value we are trying to predict.
    """
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)
    df["future_demand"] = df.groupby("product_id")["units_sold"].shift(-1)
    return df


FEATURE_COLUMNS = [
    "current_stock", "price", "discount", "supplier_lead_time", "supplier_rating",
    "promotion", "category_encoded", "season_encoded",
    "day", "month", "year", "week", "quarter", "day_of_week", "is_weekend",
    "previous_day_sales", "previous_week_sales",
    "rolling_7day_sales", "rolling_30day_sales",
    "avg_historical_demand", "sales_growth_rate", "stock_to_demand_ratio",
    "previous_month_sales", "previous_3_month_average", "previous_6_month_average",
]

TARGET_COLUMN = "future_demand"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)
    df = add_target_variable(df)

    # Drop the last row per product (no future_demand available -> can't train on it)
    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    return df
