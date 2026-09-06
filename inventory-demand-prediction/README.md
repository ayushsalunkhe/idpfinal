# StockSense — Inventory Demand Prediction and Stock Replenishment System

A complete, end-to-end Machine Learning + web application project that predicts
future product demand from historical sales data and recommends when and how
much inventory to reorder. Built as a BE Computer Engineering ML / Data
Mining project, demonstrating the full pipeline: data preprocessing → EDA →
feature engineering → model training → model comparison → evaluation →
prediction → visualization → deployment as a web app.

---

## 1. Problem Statement

Small and medium retail businesses typically reorder stock using guesswork
or simple fixed reorder levels. This leads to two costly failure modes:

- **Stockouts** — running out of a fast-moving product, losing sales and
  customer trust.
- **Overstocking** — tying up capital and warehouse space in slow-moving
  inventory that may expire or become obsolete.

The goal of this project is to replace guesswork with a data-driven system
that **learns each product's demand pattern** from historical sales and
**automatically recommends** what to reorder, how much, and how urgently.

## 2. Objectives

1. Build a realistic historical sales + inventory dataset.
2. Clean and preprocess the data (missing values, duplicates, outliers).
3. Engineer time-aware features without leaking future information.
4. Train and compare multiple regression models to predict `future_demand`.
5. Evaluate models with standard regression metrics and select the best one.
6. Convert predicted demand into an actionable **reorder recommendation**
   and a **stockout risk classification**.
7. Expose the whole system through a professional web dashboard with
   authentication, CSV upload, analytics, and downloadable reports.

## 3. Abstract

StockSense ingests multi-year, multi-product daily sales data, cleans it,
and derives a rich set of calendar, lag, and rolling-window features. Four
regression algorithms (Linear Regression, Decision Tree, Random Forest,
Gradient Boosting — plus XGBoost when the library is installed) are trained
on a **time-based** train/test split to avoid leaking future information
into the training set. The best-performing model (chosen by RMSE on the
held-out set) is saved and used at inference time to predict next-day
demand for any product. That prediction then feeds a transparent, formula-
based reorder engine (safety stock, reorder point, reorder quantity) and a
3-tier stockout-risk classifier. Everything is served through a Flask web
application backed by SQLite (users + prediction history), with analytics
pages, CSV/PDF export, and a CSV upload workflow for bringing in new data.

## 4. Existing System

Most small-business inventory tools either:
- Use a **static reorder level** ("reorder when stock < 20") that ignores
  seasonality, trends, and promotions, or
- Rely on a **manager's intuition**, which doesn't scale and isn't
  auditable, or
- Use **simple moving averages** with no ability to learn non-linear
  interactions (e.g. how a promotion during a festive weekend multiplies
  demand rather than just adding to it).

### Limitations of the existing system
- No adaptation to seasonality, promotions, or trend.
- No quantification of forecast uncertainty (safety stock is guessed).
- No historical audit trail of predictions and decisions.
- Not explainable — a manager can't see *why* a reorder was suggested.

## 5. Proposed System

StockSense proposes a machine-learning-driven pipeline that:
- Learns from **actual historical demand patterns** per product.
- Automatically compares multiple algorithms and picks the best one.
- Produces an **explainable** reorder recommendation using a documented
  formula (Reorder Point = Expected Demand During Lead Time + Safety
  Stock), where every input to that formula is itself learned or computed
  from real data (not hardcoded).
- Stores every prediction for later audit via the History page.
- Lets the business bring in their own dataset via CSV upload.

## 6. System Architecture

```
                     ┌───────────────────────────────┐
                     │        Web Browser (UI)        │
                     └───────────────┬────────────────┘
                                     │ HTTP
                     ┌───────────────▼────────────────┐
                     │     Flask App (app.py)         │
                     │  Auth · Dashboard · Upload ·   │
                     │  Prediction · Analytics ·      │
                     │  History · Reports (CSV/PDF)   │
                     └───────┬───────────────┬────────┘
                              │               │
                ┌─────────────▼───┐   ┌───────▼─────────────┐
                │  database/       │   │  ml/                 │
                │  database.py     │   │  preprocessing.py    │
                │  (SQLite: users, │   │  feature_engineering │
                │  prediction      │   │  train_model.py      │
                │  history)        │   │  evaluate_model.py    │
                └──────────────────┘   │  prediction.py        │
                                        └──────────┬────────────┘
                                                    │ loads
                                        ┌───────────▼────────────┐
                                        │  models/                │
                                        │  demand_model.pkl        │
                                        │  scaler.pkl              │
                                        │  encoders.pkl            │
                                        │  feature_columns.pkl     │
                                        └──────────────────────────┘
```

The web app **never retrains** the model during a request — training is a
separate offline step (`python ml/train_model.py`) that produces the
artifacts in `models/`, which the app loads once at startup.

## 7. Dataset Description

- File: `data/inventory_data.csv`
- ~21,900 daily records across **20 products** in **4 categories**
  (Grocery, Personal Care, Electronics, Stationery), spanning **2023–2025**.
- Generated by `data/generate_dataset.py`, which simulates realistic
  retail behaviour: festive-season demand spikes, weekend effects for
  grocery/personal-care items, random promotions with discount-driven
  demand boosts, a slow per-product trend, and Gaussian noise — plus
  deliberately injected messiness (missing values, duplicate rows, a
  few extreme outliers) so the preprocessing stage has real work to do.

| Column | Description |
|---|---|
| product_id, product_name, category | Product identity |
| date | Calendar date of the record |
| current_stock | Stock on hand at the start of the day |
| units_sold | Units actually sold that day |
| price, discount | Selling price and discount % that day |
| supplier_lead_time | Days for a reorder to arrive (per product) |
| supplier_rating | Supplier reliability score |
| season, promotion | Categorical season label; promotion flag |
| previous_month_sales, previous_3_month_average, previous_6_month_average | Historical averages, computed causally (no leakage) |

## 8. Data Preprocessing (`ml/preprocessing.py`)

| Step | Why |
|---|---|
| Date conversion | Enables chronological sorting and calendar feature extraction |
| Duplicate removal | Prevents over-weighting repeated rows during training |
| Missing-value imputation | Median (numeric, per product) / mode (categorical) — keeps rows instead of discarding valuable history |
| Outlier capping (IQR, per product) | Limits the influence of data-entry errors without breaking the daily time series (which rolling features need) |
| Categorical encoding | Label-encodes `category` and `season` for model input; encoders are saved and reused at prediction time |
| Train/test split | **Time-based**, not random — the last ~20% of the calendar is held out so evaluation reflects real forecasting conditions |

## 9. Feature Engineering (`ml/feature_engineering.py`)

Calendar features: `day, month, year, week, quarter, day_of_week, is_weekend`

Lag / rolling features (all computed using **only data strictly before**
the prediction date, per product):
`previous_day_sales, previous_week_sales, rolling_7day_sales,
rolling_30day_sales, avg_historical_demand, sales_growth_rate,
stock_to_demand_ratio`

**Target:** `future_demand` = next day's `units_sold` for that product.

**Leakage prevention:** every rolling/lag feature is built with
`shift(1)` applied *before* any rolling window or expanding aggregation,
so the row for day *T* never uses day *T*'s own sales — only information
that would genuinely have been available the day before.

## 10. ML Algorithms Used

1. **Linear Regression** — baseline linear model.
2. **Decision Tree Regressor** — captures non-linear splits, prone to overfitting alone.
3. **Random Forest Regressor** — ensemble of trees, reduces variance.
4. **Gradient Boosting Regressor** — sequential boosting, typically strongest of the four.
5. **XGBoost Regressor** — used automatically **if the `xgboost` package is
   installed**; the project runs fully and correctly without it (per the
   "XGBoost if available" requirement).

### Key formulas

**Mean Absolute Error:** MAE = (1/n) Σ |yᵢ − ŷᵢ|

**Mean Squared Error:** MSE = (1/n) Σ (yᵢ − ŷᵢ)²

**Root Mean Squared Error:** RMSE = √MSE

**R² Score:** R² = 1 − (Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²)

**Mean Absolute Percentage Error:** MAPE = (100/n) Σ |（yᵢ − ŷᵢ）/ yᵢ|  (computed over non-zero actuals only)

**Safety Stock:** SS = Z × σ_demand × √(Lead Time)
(Z = 1.65 for ~95% service level; σ_demand is each product's own
historical daily-demand standard deviation — not a hardcoded number.)

**Reorder Point:** ROP = (Predicted Daily Demand × Lead Time) + Safety Stock

**Recommended Reorder Quantity:** max(0, ROP − Current Stock)

## 11. Model Evaluation & Model Selection

`ml/train_model.py` trains every model on the same time-based training
split, evaluates all of them on the same held-out test split using
MAE/MSE/RMSE/R²/MAPE, and automatically selects the model with the
**lowest RMSE** as the production model. The comparison table and the
reasoning for the winning model are saved and displayed live on the
**Analytics** page (`ml/evaluate_model.py: explain_best_model()`).

## 12. Demand Prediction vs. Stockout Prediction

- **Demand prediction** is a *regression* problem: the trained ML model
  outputs a number — predicted units of demand for the next day.
- **Stockout prediction** is a *classification/decision* problem built on
  top of the demand prediction: given predicted demand, current stock, and
  supplier lead time, is the product at LOW / MEDIUM / HIGH risk of running
  out before the next delivery arrives? This project implements it with
  transparent, auditable business logic (see `ml/prediction.py:
  classify_stockout_risk`) rather than a second black-box model, since
  reorder decisions need to be explainable to a store manager.

## 13. Results (example run)

On a representative run, the model comparison (time-based test split) was:

| Model | RMSE | R² | MAPE % |
|---|---|---|---|
| Gradient Boosting | ~4.26 | ~0.83 | ~17.0 |
| Random Forest | ~4.51 | ~0.81 | ~18.8 |
| Linear Regression | ~4.83 | ~0.78 | ~22.3 |
| Decision Tree | ~5.48 | ~0.72 | ~20.5 |

Gradient Boosting was automatically selected as the production model.
(Exact numbers vary slightly between runs due to randomness; re-run
`python ml/train_model.py` to regenerate `models/model_comparison.csv`.)

## 14. Screenshots

_Add screenshots of the following pages here before your presentation:_
- Login / Signup
- Dashboard (KPIs + recent predictions)
- Upload Dataset (with validation)
- Prediction page (result panel)
- Analytics page (model comparison + charts)
- History page (with CSV/PDF export)

## 15. Installation

```bash
# 1. Clone / extract the project, then move into it
cd inventory-demand-prediction

# 2. Install dependencies
pip install -r requirements.txt
# NOTE: xgboost is optional. If it fails to install in your environment,
# remove it from requirements.txt — the project runs fully without it.
```

## 16. How to Run

```bash
# Step 1 (only needed once, or after uploading a new dataset):
# train all models, compare them, and save the best one + evaluation graphs
python ml/train_model.py

# Step 2: start the web app
python app.py
```

Then open **http://127.0.0.1:5000** in your browser, create an account, and
sign in.

## 17. Project Structure

```
inventory-demand-prediction/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── inventory_data.csv
│   └── generate_dataset.py
├── models/                  (created by train_model.py)
│   ├── demand_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── feature_columns.pkl
│   ├── model_meta.json
│   └── model_comparison.csv
├── ml/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── prediction.py
├── database/
│   └── database.py
├── templates/
│   ├── login.html, signup.html, base.html
│   ├── dashboard.html, upload.html, prediction.html
│   ├── analytics.html, history.html
├── static/
│   ├── css/style.css, js/main.js
│   └── images/  (evaluation graphs, generated by training)
├── docs/
│   └── presentation_content.md
└── reports/                 (ad-hoc export location; live downloads are streamed, not stored here)
```

## 18. Future Scope

- Add SHAP-based explainability alongside built-in feature importance.
- Support multi-step (7/30-day) demand forecasting instead of next-day only.
- Add supplier-side integration (auto-generate purchase orders).
- Add per-category or per-store models for chains with heterogeneous demand.
- Move from SQLite to PostgreSQL and add role-based access (manager vs. staff).

## 19. Conclusion

StockSense demonstrates a complete, leakage-safe ML pipeline for demand
forecasting, wrapped in a real, usable web application. It goes beyond a
rule-based calculator: it genuinely trains, compares, evaluates, and
deploys multiple regression models, and uses the ML model's own output as
the core input to an explainable, formula-based reorder and stockout-risk
system — the same architecture pattern used in real-world inventory
management platforms.
