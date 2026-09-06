import sqlite3
from pathlib import Path

db = Path(__file__).resolve().parents[1] / 'database' / 'inventory_system.db'
conn = sqlite3.connect(str(db))
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM prediction_history')
total = cur.fetchone()[0]
cur.execute('SELECT COUNT(DISTINCT product_id) FROM prediction_history')
distinct = cur.fetchone()[0]
cur.execute('SELECT product_id, product_name, COUNT(*) as c FROM prediction_history GROUP BY product_id,product_name ORDER BY c DESC LIMIT 20')
top = cur.fetchall()
cur.execute("SELECT product_id, product_name, created_at FROM prediction_history ORDER BY created_at DESC LIMIT 20")
recent = cur.fetchall()
conn.close()
print('total_rows=', total)
print('distinct_product_ids=', distinct)
print('\nTop product counts (product_id, product_name, count):')
for r in top:
    print(r)
print('\nMost recent entries (product_id, product_name, created_at):')
for r in recent:
    print(r)
