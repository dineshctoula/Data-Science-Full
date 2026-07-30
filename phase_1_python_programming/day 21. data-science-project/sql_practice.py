import sqlite3

conn = sqlite3.connect("housing.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE housing (
    income REAL,
    age INTEGER,
    rooms INTEGER,
    price REAL
)
""")