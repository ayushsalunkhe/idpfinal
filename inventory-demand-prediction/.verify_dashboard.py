import sqlite3
from werkzeug.security import generate_password_hash
from app import app
from database import database as db

conn = sqlite3.connect(db.DB_PATH)
cur = conn.cursor()
cur.execute("DELETE FROM prediction_history")
cur.execute("DELETE FROM users")
cur.execute("INSERT INTO users (id, username, email, password_hash, created_at) VALUES (?, ?, ?, ?, datetime('now'))", (1, "alice", "alice@test.com", generate_password_hash("pass123")))
cur.execute("INSERT INTO users (id, username, email, password_hash, created_at) VALUES (?, ?, ?, ?, datetime('now'))", (2, "bob", "bob@test.com", generate_password_hash("pass123")))
cur.execute("INSERT INTO prediction_history (user_id, product_id, product_name, category, predicted_demand, current_stock, safety_stock, reorder_point, recommended_reorder_qty, stockout_risk, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))", (1, 101, "Alpha Widget", "A", 25, 10, 5, 15, 8, "LOW RISK", "SUFFICIENT"))
cur.execute("INSERT INTO prediction_history (user_id, product_id, product_name, category, predicted_demand, current_stock, safety_stock, reorder_point, recommended_reorder_qty, stockout_risk, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))", (2, 202, "Beta Widget", "B", 40, 8, 3, 12, 10, "HIGH RISK", "REORDER REQUIRED"))
conn.commit()
conn.close()

client = app.test_client()
resp1 = client.post('/login', data={'username':'alice','password':'pass123'}, follow_redirects=True)
assert resp1.status_code == 200
page1 = resp1.get_data(as_text=True)
assert 'Alpha Widget' in page1 and 'Beta Widget' not in page1
with client.session_transaction() as sess:
    sess.clear()
resp2 = client.post('/login', data={'username':'bob','password':'pass123'}, follow_redirects=True)
assert resp2.status_code == 200
page2 = resp2.get_data(as_text=True)
assert 'Beta Widget' in page2 and 'Alpha Widget' not in page2
print('Dashboard isolation verified for Alice and Bob')
print('Alice stats', db.get_dashboard_stats(1))
print('Bob stats', db.get_dashboard_stats(2))
