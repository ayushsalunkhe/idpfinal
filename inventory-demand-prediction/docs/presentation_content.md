# Presentation Content — StockSense: Inventory Demand Prediction and Stock Replenishment System

_Use this as the script/outline for slides or a viva. Each section below maps to one or two slides._

---

### 1. Title Slide
**StockSense — Inventory Demand Prediction and Stock Replenishment System**
A Machine Learning approach to demand forecasting and reorder automation.

### 2. Introduction
Retailers of every size need to answer one recurring question: *how much of
each product should we have on hand, and when should we reorder more?*
StockSense answers this using historical sales data and machine learning,
rather than fixed rules or guesswork.

### 3. Problem Statement
Manual or rule-based inventory management cannot adapt to seasonality,
promotions, or shifting demand trends, leading to stockouts (lost sales)
and overstocking (locked-up capital). There is a need for a system that
learns each product's real demand pattern and translates that into a
concrete, explainable reorder decision.

### 4. Objectives
- Predict next-period product demand using historical sales data.
- Compare multiple regression algorithms and automatically select the best.
- Convert demand predictions into a stockout-risk classification.
- Recommend a specific, justified reorder quantity.
- Package the system as a usable web application.

### 5. Literature Review (talking points)
- Traditional inventory control (EOQ, fixed reorder point) assumes stable,
  known demand — unrealistic for real retail.
- Time-series methods (moving average, exponential smoothing) capture
  trend/seasonality but not interaction effects (e.g. promotion × weekend).
- Tree-based ML ensembles (Random Forest, Gradient Boosting) have become
  the standard for tabular demand-forecasting competitions (e.g. Kaggle
  M5 Forecasting) because they model non-linear feature interactions well.

### 6. Existing System
- Static reorder levels or manager intuition.
- No seasonality/promotion awareness.
- No forecast uncertainty quantification (safety stock is guessed).
- Not explainable or auditable.

### 7. Limitations of Existing System
- Reacts to stockouts rather than anticipating them.
- Cannot scale across hundreds of SKUs.
- No historical record of *why* a decision was made.

### 8. Proposed System
An ML pipeline that:
1. Cleans and engineers features from historical sales data.
2. Trains and compares 4–5 regression algorithms.
3. Selects the best model automatically (lowest RMSE).
4. Feeds predicted demand into a transparent reorder-point formula.
5. Classifies stockout risk into LOW / MEDIUM / HIGH.
6. Serves everything through a login-protected web dashboard.

### 9. Methodology (pipeline diagram)
Raw CSV → Cleaning → Feature Engineering → Time-based Train/Test Split →
Train (Linear Regression, Decision Tree, Random Forest, Gradient Boosting,
XGBoost*) → Evaluate (MAE, MSE, RMSE, R², MAPE) → Select Best Model →
Save Model → Predict → Reorder Engine → Web Dashboard
<br>*XGBoost used automatically if installed.

### 10. System Architecture
Browser → Flask app (`app.py`) → ML layer (`ml/`) loads a pre-trained model
from `models/` → SQLite database (`database/`) stores users and prediction
history. Training (`ml/train_model.py`) is a separate offline step.

### 11. Dataset
~21,900 daily records, 20 products, 4 categories, 3 years (2023–2025),
15 core columns including price, discount, supplier lead time, season, and
historical rolling averages. Synthetically generated with realistic
seasonality, weekend effects, promotions, trend, and injected data-quality
issues (missing values, duplicates, outliers) to mirror real-world data.

### 12. Data Preprocessing
Date parsing, duplicate removal, per-product median/mode imputation,
per-product IQR-based outlier capping, label encoding of categorical
columns, and a **time-based** (not random) train/test split to avoid
information leakage from the future into training.

### 13. Feature Engineering
Calendar features (day/month/year/week/quarter/weekend), lag features
(previous day/week sales), rolling averages (7-day, 30-day), expanding
historical average, sales growth rate, and stock-to-demand ratio — all
computed causally (using `shift()` before any rolling/aggregation) so no
feature ever "sees the future."

### 14. Algorithms Used
Linear Regression, Decision Tree Regressor, Random Forest Regressor,
Gradient Boosting Regressor, and XGBoost Regressor (if available). Trained
on identical splits so the comparison is fair.

### 15. Model Training
`ml/train_model.py` orchestrates the entire pipeline end-to-end and is
re-runnable any time new data is uploaded through the web app.

### 16. Results
Gradient Boosting achieved the best performance in testing (lowest RMSE,
highest R²), outperforming Linear Regression because retail demand depends
on non-linear interactions (e.g. promotion effects compound with festive
seasonality) that linear models cannot capture. Full comparison table is
generated in `models/model_comparison.csv` and shown live on the Analytics
page.

### 17. Advantages
- Explainable: every reorder number traces back to a documented formula.
- Adapts automatically to each product's own demand variability (safety
  stock uses that product's real historical standard deviation).
- Extensible: new data can be uploaded and the model retrained without
  code changes.
- Full audit trail via prediction history.

### 18. Applications
- Kirana / supermarket chains, pharmacies, hardware stores, e-commerce
  warehouses — any business managing multi-SKU inventory with recurring
  demand.

### 19. Future Scope
SHAP explainability, multi-step forecasting horizons, supplier-side
purchase-order automation, per-store models, and production-grade database
migration.

### 20. Conclusion
StockSense shows that a genuinely trained, compared, and evaluated ML
pipeline — not a disguised rule-based calculator — can power a practical,
explainable inventory replenishment system suitable for real deployment
and for demonstrating core ML/Data Mining concepts.

### 21. References
- Hyndman, R.J., Athanasopoulos, G. — *Forecasting: Principles and
  Practice*
- Scikit-learn documentation — https://scikit-learn.org
- XGBoost documentation — https://xgboost.readthedocs.io
- Silver, E.A., Pyke, D.F., Peterson, R. — *Inventory Management and
  Production Planning and Scheduling* (safety stock / reorder point theory)
