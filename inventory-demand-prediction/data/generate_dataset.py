"""
generate_dataset.py
--------------------
Generates a realistic synthetic inventory/sales dataset for the
Inventory Demand Prediction and Stock Replenishment System.

Why synthetic data?
A real retail chain would never publish raw sales + stock data (privacy /
competitive reasons), so for a college project we simulate a dataset that
has the same statistical *shape* as real retail data:
    - seasonality (festive months sell more)
    - weekday/weekend effects
    - promotions temporarily boosting sales
    - trend + noise
    - category-level differences (e.g. groceries sell more units than
      electronics, but electronics have higher price)

This keeps the ML problem realistic: the model actually has to learn
patterns instead of memorizing a rule.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# ---------------------------------------------------------------------
# 1. Define the product catalog
# ---------------------------------------------------------------------
PRODUCTS = [
    # (product_id, product_name,        category,       base_price, base_daily_demand)
    (1,  "Rice 5kg",                "Grocery",       320,  18),
    (2,  "Wheat Flour 5kg",         "Grocery",       260,  15),
    (3,  "Sunflower Oil 1L",        "Grocery",       165,  20),
    (4,  "Toor Dal 1kg",            "Grocery",       140,  12),
    (5,  "Sugar 1kg",               "Grocery",        48,  22),
    (6,  "Tea Powder 500g",         "Grocery",       210,  10),
    (7,  "Salt 1kg",                "Grocery",        22,  25),
    (8,  "Basmati Rice 1kg",        "Grocery",       120,   9),
    (9,  "Toothpaste 150g",         "Personal Care",  95,  14),
    (10, "Shampoo 340ml",           "Personal Care", 210,   8),
    (11, "Soap Bar (Pack of 4)",    "Personal Care", 160,  16),
    (12, "Hand Sanitizer 500ml",    "Personal Care", 130,   7),
    (13, "LED Bulb 9W",             "Electronics",     90,   6),
    (14, "Extension Board",         "Electronics",    350,   3),
    (15, "USB Cable Type-C",        "Electronics",    250,   5),
    (16, "Bluetooth Earphones",     "Electronics",    999,   2),
    (17, "Notebook 200 pages",      "Stationery",      45,  20),
    (18, "Ball Pen (Pack of 5)",    "Stationery",      50,  18),
    (19, "A4 Paper Ream",           "Stationery",     260,   6),
    (20, "Whiteboard Marker Set",   "Stationery",     150,   5),
]

CATEGORY_SUPPLIER = {
    "Grocery":       (3, 7, 4.2),      # (min_lead, max_lead, avg_supplier_rating)
    "Personal Care": (5, 10, 4.0),
    "Electronics":   (7, 15, 4.5),
    "Stationery":    (4, 9, 3.8),
}

START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2025, 12, 31)
DATE_RANGE = pd.date_range(START_DATE, END_DATE, freq="D")

FESTIVE_MONTHS = {10, 11, 3}   # Diwali/festive season + March (year-end sales)
SUMMER_MONTHS = {4, 5, 6}

records = []

for pid, pname, category, base_price, base_demand in PRODUCTS:
    lead_min, lead_max, supplier_rating_avg = CATEGORY_SUPPLIER[category]
    supplier_lead_time = np.random.randint(lead_min, lead_max + 1)
    supplier_rating = round(np.clip(np.random.normal(supplier_rating_avg, 0.3), 2.5, 5.0), 1)

    # Slow long-term trend: some products trend up, some down slightly
    trend_slope = np.random.uniform(-0.002, 0.004)

    # Running "current stock" simulation
    current_stock = int(base_demand * np.random.uniform(15, 30))

    stock_history = []
    sales_history = []

    for i, date in enumerate(DATE_RANGE):
        month = date.month
        weekday = date.weekday()  # 0=Mon ... 6=Sun

        # ---- seasonality ----
        season_factor = 1.0
        if month in FESTIVE_MONTHS:
            season_factor = np.random.uniform(1.3, 1.8)
            season = "Festive"
        elif month in SUMMER_MONTHS:
            season_factor = np.random.uniform(0.9, 1.1)
            season = "Summer"
        else:
            season_factor = np.random.uniform(0.85, 1.05)
            season = "Regular"

        # ---- weekend boost for groceries/personal care ----
        weekend_factor = 1.25 if weekday >= 5 and category in ("Grocery", "Personal Care") else 1.0

        # ---- promotion (random ~8% of days) ----
        promotion = 1 if np.random.rand() < 0.08 else 0
        promo_factor = np.random.uniform(1.4, 2.0) if promotion else 1.0
        discount = round(np.random.uniform(10, 30), 1) if promotion else round(np.random.uniform(0, 5), 1)

        # ---- trend + noise ----
        trend_factor = 1 + trend_slope * i / 30
        noise = np.random.normal(1.0, 0.12)

        demand_today = max(
            0,
            base_demand * season_factor * weekend_factor * promo_factor * trend_factor * noise
        )
        units_sold = int(round(demand_today))

        # price fluctuates slightly + drops during discount
        price = round(base_price * (1 - discount / 100) * np.random.uniform(0.98, 1.02), 2)

        # ---- stock dynamics ----
        current_stock = max(current_stock - units_sold, 0)
        # Reorder simulation: if stock runs low, a delivery arrives after lead time
        if current_stock < base_demand * 5:
            current_stock += int(base_demand * np.random.uniform(20, 35))

        sales_history.append(units_sold)

        # previous month / 3-month / 6-month averages (using history so far -> no leakage)
        def avg_last_n_days(hist, n):
            if len(hist) == 0:
                return 0
            window = hist[-n:]
            return round(sum(window) / len(window), 2)

        previous_month_sales = avg_last_n_days(sales_history[:-1], 30) if i > 0 else 0
        previous_3_month_average = avg_last_n_days(sales_history[:-1], 90) if i > 0 else 0
        previous_6_month_average = avg_last_n_days(sales_history[:-1], 180) if i > 0 else 0

        records.append({
            "product_id": pid,
            "product_name": pname,
            "category": category,
            "date": date.strftime("%Y-%m-%d"),
            "current_stock": current_stock,
            "units_sold": units_sold,
            "price": price,
            "discount": discount,
            "supplier_lead_time": supplier_lead_time,
            "supplier_rating": supplier_rating,
            "season": season,
            "promotion": promotion,
            "previous_month_sales": previous_month_sales,
            "previous_3_month_average": previous_3_month_average,
            "previous_6_month_average": previous_6_month_average,
        })

df = pd.DataFrame(records)

# Inject a small amount of realistic messiness so preprocessing has real work to do
# 1) A few missing values
missing_idx = np.random.choice(df.index, size=int(0.01 * len(df)), replace=False)
df.loc[missing_idx, "supplier_rating"] = np.nan

missing_idx2 = np.random.choice(df.index, size=int(0.005 * len(df)), replace=False)
df.loc[missing_idx2, "current_stock"] = np.nan

# 2) A handful of duplicate rows
dupes = df.sample(n=25, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

# 3) A few extreme outliers in units_sold (data entry errors)
outlier_idx = np.random.choice(df.index, size=15, replace=False)
df.loc[outlier_idx, "units_sold"] = df.loc[outlier_idx, "units_sold"] * np.random.randint(15, 25)

df = df.sample(frac=1, random_state=7).reset_index(drop=True)  # shuffle rows

OUT_PATH = "data/inventory_data.csv"
df.to_csv(OUT_PATH, index=False)
print(f"Dataset generated: {OUT_PATH}")
print(f"Total records: {len(df)}")
print(df.head())
