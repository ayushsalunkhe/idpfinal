"""
database.py
------------
SQLite persistence layer for:
    - users (registration / login, password-hashed)
    - prediction_history (every prediction the app has ever made)
    - uploaded_datasets (metadata about CSVs uploaded by users)

Using SQLite (not a heavier DB) is appropriate here since this is a
single-node college / demo project; the same schema would map cleanly to
Postgres/MySQL in a production deployment.
"""

import sqlite3
import os
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(__file__), "inventory_system.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they do not already exist."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            product_id INTEGER,
            product_name TEXT,
            category TEXT,
            predicted_demand REAL,
            current_stock REAL,
            safety_stock REAL,
            reorder_point REAL,
            recommended_reorder_qty REAL,
            stockout_risk TEXT,
            status TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS uploaded_datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT,
            row_count INTEGER,
            uploaded_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------
def create_user(username: str, email: str, password: str) -> tuple[bool, str]:
    """Hash the password (never store plaintext) and insert a new user."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        password_hash = generate_password_hash(password)
        cur.execute(
            "INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (username, email, password_hash, datetime.now().isoformat())
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "Username or email already exists."
    finally:
        conn.close()


def verify_user(username_or_email: str, password: str):
    """Return the user row if credentials are valid, else None."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM users WHERE username = ? OR email = ?",
        (username_or_email, username_or_email)
    )
    user = cur.fetchone()
    conn.close()

    if user and check_password_hash(user["password_hash"], password):
        return user
    return None


def get_user_by_id(user_id: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cur.fetchone()
    conn.close()
    return user


def get_user_upload_count(user_id: int) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as c FROM uploaded_datasets WHERE user_id = ?", (user_id,))
    count = cur.fetchone()["c"]
    conn.close()
    return int(count or 0)


# ---------------------------------------------------------------------
# Prediction history
# ---------------------------------------------------------------------
def save_prediction(user_id: int, result: dict):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO prediction_history
        (user_id, product_id, product_name, category, predicted_demand, current_stock,
         safety_stock, reorder_point, recommended_reorder_qty, stockout_risk, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, result.get("product_id"), result.get("product_name"), result.get("category"),
        result.get("predicted_demand"), result.get("current_stock"), result.get("safety_stock"),
        result.get("reorder_point"), result.get("recommended_reorder_qty"),
        result.get("stockout_risk"), result.get("status"), datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()


def get_prediction_history(user_id: int, limit: int = 100):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM prediction_history WHERE user_id = ?
        ORDER BY created_at DESC LIMIT ?
    """, (user_id, limit))
    rows = cur.fetchall()
    conn.close()
    return rows


def clear_user_history(user_id: int) -> int:
    """Delete all saved predictions for a single user and return row count removed."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM prediction_history WHERE user_id = ?", (user_id,))
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    return int(deleted or 0)


def get_recent_predictions_all(limit: int = 10):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM prediction_history ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_recent_predictions_for_user(user_id: int, limit: int = 10):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM prediction_history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def get_dashboard_stats(user_id: int | None = None):
    """Aggregate stats for the dashboard page.

    When user_id is provided, the stats are scoped to the signed-in user;
    otherwise they reflect the full app history for the current global view.
    """
    conn = get_connection()
    cur = conn.cursor()

    where_clause = ""
    params = []
    if user_id is not None:
        where_clause = "WHERE p.user_id = ?"
        params.append(user_id)

    # Use only the latest prediction per product to avoid counting duplicate
    # historical rows. We select the most recent row for each product_id
    # then aggregate over that reduced set.
    cur.execute(f"""
        WITH latest AS (
            SELECT product_id, MAX(created_at) AS max_created
            FROM prediction_history
            GROUP BY product_id
        )
        SELECT COUNT(*) as c
        FROM prediction_history p
        JOIN latest l ON p.product_id = l.product_id AND p.created_at = l.max_created
        {where_clause}
    """, params)
    total_predictions = cur.fetchone()[0]

    cur.execute(f"""
        WITH latest AS (
            SELECT product_id, MAX(created_at) AS max_created
            FROM prediction_history
            GROUP BY product_id
        )
        SELECT COUNT(*) as c
        FROM prediction_history p
        JOIN latest l ON p.product_id = l.product_id AND p.created_at = l.max_created
        {where_clause}
        AND p.status = 'REORDER REQUIRED'
    """, params)
    reorder_needed = cur.fetchone()[0]

    cur.execute(f"""
        WITH latest AS (
            SELECT product_id, MAX(created_at) AS max_created
            FROM prediction_history
            GROUP BY product_id
        )
        SELECT COUNT(*) as c
        FROM prediction_history p
        JOIN latest l ON p.product_id = l.product_id AND p.created_at = l.max_created
        {where_clause}
        AND p.stockout_risk = 'HIGH RISK'
    """, params)
    high_risk = cur.fetchone()[0]

    cur.execute(f"""
        WITH latest AS (
            SELECT product_id, MAX(created_at) AS max_created
            FROM prediction_history
            GROUP BY product_id
        )
        SELECT AVG(p.predicted_demand) as a
        FROM prediction_history p
        JOIN latest l ON p.product_id = l.product_id AND p.created_at = l.max_created
        {where_clause}
    """, params)
    avg_row = cur.fetchone()
    avg_demand = round(avg_row[0], 2) if avg_row and avg_row[0] is not None else 0

    conn.close()
    return {
        "total_predictions": total_predictions,
        "reorder_needed": reorder_needed,
        "high_risk": high_risk,
        "avg_predicted_demand": avg_demand,
    }


def record_upload(user_id: int, filename: str, row_count: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO uploaded_datasets (user_id, filename, row_count, uploaded_at) VALUES (?, ?, ?, ?)",
        (user_id, filename, row_count, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
