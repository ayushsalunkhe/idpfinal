"""
app.py
-------
Flask backend for the Inventory Demand Prediction and Stock
Replenishment System.

Routes:
    /signup, /login, /logout        -> authentication
    /dashboard                      -> summary KPIs
    /upload                         -> CSV dataset upload + validation
    /prediction                     -> per-product demand/reorder prediction
    /analytics                      -> model performance & trend charts
    /history                        -> past predictions (per user)
    /reports/csv, /reports/pdf      -> downloadable reports
    /api/chart/<name>               -> JSON data for interactive Chart.js charts

The app loads the ALREADY-TRAINED model (models/demand_model.pkl) at
request time -- it never retrains during a web request. Run
`python ml/train_model.py` once beforehand (see README).
"""

import os
import io
import csv
import json
from datetime import datetime
from functools import wraps

import numpy as np
import pandas as pd
from flask import (Flask, render_template, request, redirect, url_for,
                    session, flash, send_file, jsonify)

from database import database as db
from ml import preprocessing, feature_engineering, prediction as pred_module, evaluate_model

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "inventory_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

db.init_db()

# ---------------------------------------------------------------------
# In-memory cache of the feature-engineered dataframe.
# Rebuilt only on startup or after a new dataset is uploaded, since
# cleaning + feature engineering the full history on every request
# would be wasteful.
# ---------------------------------------------------------------------
_FEATURE_DATA_CACHE = {"df": None}


def get_feature_data(force_reload: bool = False) -> pd.DataFrame:
    if _FEATURE_DATA_CACHE["df"] is None or force_reload:
        df, _encoders = preprocessing.clean_pipeline(DATA_PATH)
        df = feature_engineering.build_features(df)
        _FEATURE_DATA_CACHE["df"] = df
    return _FEATURE_DATA_CACHE["df"]


def model_is_trained() -> bool:
    return os.path.exists(os.path.join(MODELS_DIR, "demand_model.pkl"))


def user_has_uploaded_dataset(user_id: int) -> bool:
    return db.get_user_upload_count(user_id) > 0


# ---------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


@app.context_processor
def inject_globals():
    return {
        "current_username": session.get("username"),
        "model_trained": model_is_trained(),
    }


# ---------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not username or not email or not password:
            flash("All fields are required.", "danger")
        elif password != confirm:
            flash("Passwords do not match.", "danger")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
        else:
            success, message = db.create_user(username, email, password)
            flash(message, "success" if success else "danger")
            if success:
                return redirect(url_for("login"))
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        identifier = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = db.verify_user(identifier, password)
        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash(f"Welcome back, {user['username']}!", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid username/email or password.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# ---------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    user_id = session["user_id"]
    if not user_has_uploaded_dataset(user_id):
        empty_stats = {
            "total_predictions": 0,
            "reorder_needed": 0,
            "high_risk": 0,
            "avg_predicted_demand": 0,
        }
        return render_template(
            "dashboard.html",
            stats=empty_stats,
            recent=[],
            total_products=0,
            total_inventory=0,
            avg_price=0,
        )

    stats = db.get_dashboard_stats(user_id)
    recent = db.get_recent_predictions_for_user(user_id, 6)

    total_products, total_inventory, avg_price = 0, 0, 0
    if model_is_trained():
        df = get_feature_data()
        total_products = int(df["product_id"].nunique())
        latest_stock = df.sort_values("date").groupby("product_id")["current_stock"].last()
        total_inventory = int(latest_stock.sum())
        avg_price = round(float(df["price"].mean()), 2)

    return render_template(
        "dashboard.html",
        stats=stats,
        recent=recent,
        total_products=total_products,
        total_inventory=total_inventory,
        avg_price=avg_price,
    )


# ---------------------------------------------------------------------
# Upload dataset
# ---------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "product_id", "product_name", "category", "date", "current_stock",
    "units_sold", "price", "discount", "supplier_lead_time", "supplier_rating",
    "season", "promotion", "previous_month_sales",
]
OPTIONAL_HISTORY_COLUMNS = ["previous_3_month_average", "previous_6_month_average"]


def _ensure_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill missing historical summary columns from each product's prior sales history."""
    df = df.copy()
    sorted_df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

    for col in ["previous_month_sales", *OPTIONAL_HISTORY_COLUMNS]:
        if col not in sorted_df.columns:
            sorted_df[col] = np.nan

    def rolling_average(series: pd.Series, window_days: int) -> pd.Series:
        values = []
        for idx, value in enumerate(series):
            window_start = max(0, idx - window_days)
            window = series.iloc[window_start:idx]
            values.append(float(window.mean()) if len(window) else 0.0)
        return pd.Series(values, index=series.index)

    for product_id, product_df in sorted_df.groupby("product_id", sort=False):
        product_idx = product_df.index
        product_df = product_df.sort_values("date")

        for target_col, window_days in {
            "previous_month_sales": 30,
            "previous_3_month_average": 90,
            "previous_6_month_average": 180,
        }.items():
            sorted_df.loc[product_idx, target_col] = rolling_average(
                product_df[target_col].fillna(product_df["units_sold"]), window_days
            ).values

    return sorted_df


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        file = request.files.get("dataset_file")
        if not file or file.filename == "":
            flash("Please choose a CSV file to upload.", "danger")
            return redirect(url_for("upload"))

        if not file.filename.lower().endswith(".csv"):
            flash("Only .csv files are supported.", "danger")
            return redirect(url_for("upload"))

        try:
            new_df = pd.read_csv(file)
        except Exception as e:
            flash(f"Could not read the CSV file: {e}", "danger")
            return redirect(url_for("upload"))

        missing_cols = [c for c in REQUIRED_COLUMNS if c not in new_df.columns]
        if missing_cols:
            flash(
                "Upload failed. The CSV is missing required column(s): "
                + ", ".join(missing_cols)
                + f". Required columns are: {', '.join(REQUIRED_COLUMNS)}",
                "danger",
            )
            return redirect(url_for("upload"))

        if any(col not in new_df.columns for col in OPTIONAL_HISTORY_COLUMNS):
            new_df = _ensure_history_columns(new_df)

        for col in ["previous_month_sales", *OPTIONAL_HISTORY_COLUMNS]:
            if col in new_df.columns:
                new_df[col] = pd.to_numeric(new_df[col], errors="coerce").fillna(0)

        new_df.to_csv(DATA_PATH, index=False)
        db.record_upload(session["user_id"], file.filename, len(new_df))
        get_feature_data(force_reload=True)

        flash(
            f"Dataset uploaded successfully ({len(new_df)} rows). "
            "Note: retrain the model (python ml/train_model.py) so predictions "
            "reflect the new data.",
            "success",
        )
        return redirect(url_for("upload"))

    return render_template("upload.html", required_columns=REQUIRED_COLUMNS)


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------
@app.route("/prediction", methods=["GET", "POST"])
@login_required
def prediction_page():
    if not user_has_uploaded_dataset(session["user_id"]):
        flash("Upload a CSV dataset before generating predictions.", "warning")
        return render_template("prediction.html", products=[], result=None)

    if not model_is_trained():
        flash("No trained model found. Please run: python ml/train_model.py", "warning")
        return render_template("prediction.html", products=[], result=None)

    df = get_feature_data()
    products = (
        df[["product_id", "product_name", "category"]]
        .drop_duplicates()
        .sort_values("product_name")
        .to_dict("records")
    )

    result = None
    results = []
    if request.method == "POST":
        # If user clicked "Predict all products"
        if request.form.get("predict_all"):
            try:
                products_to_predict = df["product_id"].drop_duplicates().tolist()
                count = 0
                results = []
                for pid in products_to_predict:
                    try:
                        res = pred_module.predict_for_product(df, pid)
                        db.save_prediction(session["user_id"], res)
                        results.append(res)
                        count += 1
                    except Exception:
                        # skip individual failures and continue
                        continue
                result = results[0] if results else None
                flash(f"Predictions generated and saved for {count} products.", "success")
            except Exception as e:
                flash(f"Could not generate predictions: {e}", "danger")
        else:
            try:
                raw_id = request.form.get("product_id")
                try:
                    # prefer numeric id if possible
                    product_id = int(raw_id)
                except (TypeError, ValueError):
                    product_id = raw_id
                result = pred_module.predict_for_product(df, product_id)
                db.save_prediction(session["user_id"], result)
                flash("Prediction generated and saved to history.", "success")
            except Exception as e:
                flash(f"Could not generate prediction: {e}", "danger")

    return render_template("prediction.html", products=products, result=result, results=results)


# ---------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------
@app.route("/analytics")
@login_required
def analytics():
    comparison_df = evaluate_model.load_model_comparison()
    meta = evaluate_model.load_model_meta()
    explanation = evaluate_model.explain_best_model()

    return render_template(
        "analytics.html",
        comparison=comparison_df.to_dict("records") if not comparison_df.empty else [],
        meta=meta,
        explanation=explanation,
    )


@app.route("/api/chart/monthly-trend")
@login_required
def api_monthly_trend():
    df = get_feature_data()
    monthly = df.copy()
    monthly["year_month"] = monthly["date"].dt.to_period("M").astype(str)
    grouped = monthly.groupby("year_month")["units_sold"].sum().reset_index()
    return jsonify({
        "labels": grouped["year_month"].tolist(),
        "values": grouped["units_sold"].round(1).tolist(),
    })


@app.route("/api/chart/category-performance")
@login_required
def api_category_performance():
    df = get_feature_data()
    grouped = df.groupby("category")["units_sold"].sum().sort_values(ascending=False).reset_index()
    return jsonify({
        "labels": grouped["category"].tolist(),
        "values": grouped["units_sold"].round(1).tolist(),
    })


@app.route("/api/chart/product-demand")
@login_required
def api_product_demand():
    df = get_feature_data()
    grouped = (
        df.groupby("product_name")["units_sold"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    return jsonify({
        "labels": grouped["product_name"].tolist(),
        "values": grouped["units_sold"].round(1).tolist(),
    })


# ---------------------------------------------------------------------
# Prediction history
# ---------------------------------------------------------------------
@app.route("/history")
@login_required
def history():
    rows = db.get_prediction_history(session["user_id"])
    return render_template("history.html", rows=rows)


@app.route("/history/clear", methods=["POST"])
@login_required
def clear_history():
    deleted = db.clear_user_history(session["user_id"])
    if deleted:
        flash(f"Cleared {deleted} saved prediction(s).", "success")
    else:
        flash("No saved predictions to clear.", "info")
    return redirect(url_for("history"))


# ---------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------
@app.route("/reports/csv")
@login_required
def download_csv():
    rows = db.get_prediction_history(session["user_id"], limit=1000)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Date", "Product", "Category", "Predicted Demand", "Current Stock",
                      "Safety Stock", "Reorder Point", "Recommended Reorder Qty",
                      "Stockout Risk", "Status"])
    for r in rows:
        writer.writerow([r["created_at"], r["product_name"], r["category"], r["predicted_demand"],
                          r["current_stock"], r["safety_stock"], r["reorder_point"],
                          r["recommended_reorder_qty"], r["stockout_risk"], r["status"]])

    mem = io.BytesIO(output.getvalue().encode("utf-8"))
    return send_file(mem, mimetype="text/csv", as_attachment=True,
                      download_name=f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")


@app.route("/reports/pdf")
@login_required
def download_pdf():
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm

    rows = db.get_prediction_history(session["user_id"], limit=200)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4))
    styles = getSampleStyleSheet()
    elements = [Paragraph("Inventory Demand Prediction Report", styles["Title"]),
                Spacer(1, 0.5 * cm),
                Paragraph(f"Generated for user: {session.get('username')}", styles["Normal"]),
                Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]),
                Spacer(1, 0.5 * cm)]

    table_data = [["Date", "Product", "Category", "Pred. Demand", "Stock",
                   "Safety Stock", "Reorder Pt.", "Reorder Qty", "Risk", "Status"]]
    for r in rows:
        table_data.append([
            r["created_at"][:16], r["product_name"], r["category"],
            r["predicted_demand"], r["current_stock"], r["safety_stock"],
            r["reorder_point"], r["recommended_reorder_qty"], r["stockout_risk"], r["status"],
        ])

    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)

    return send_file(buffer, mimetype="application/pdf", as_attachment=True,
                      download_name=f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
