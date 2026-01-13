import sqlite3

db_path = "chroma.sqlite3"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type='table';
""")

tables = cursor.fetchall()

for (table_name,) in tables:
    print(f"\nSchema for table: {table_name}")
    cursor.execute(f"PRAGMA table_info({table_name});")
    for column in cursor.fetchall():
        print(column)

conn.close()
