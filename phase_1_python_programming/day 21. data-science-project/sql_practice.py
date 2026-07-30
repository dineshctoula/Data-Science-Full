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


cursor.executemany("""
INSERT INTO housing VALUES (?, ?, ?, ?)
""", [
    (50000, 10, 3, 200000),
    (60000, 5, 4, 250000),
    (45000, 20, 2, 150000)
])

conn.commit()