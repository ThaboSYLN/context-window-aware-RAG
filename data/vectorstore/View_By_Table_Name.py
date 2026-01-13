import sqlite3

db_path = "chroma.sqlite3"
table_name = "TABLE_NAME_HERE"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()
