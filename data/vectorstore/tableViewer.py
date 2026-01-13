import sqlite3

db_path = "chroma.sqlite3"  # update path if needed

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type='table';
""")

tables = cursor.fetchall()

print("Tables in chroma.sqlite3:")
for table in tables:
    print(f"- {table[0]}")

conn.close()
