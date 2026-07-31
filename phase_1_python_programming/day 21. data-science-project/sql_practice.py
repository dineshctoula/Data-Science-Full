import sqlite3

# Connect to database (creates file if it doesn't exist)
conn = sqlite3.connect("housing.db")
cursor = conn.cursor()

# 1. Create table safely (prevents crashing on repeat runs)
cursor.execute("""
CREATE TABLE IF NOT EXISTS housing (
    income REAL,
    age INTEGER,
    rooms INTEGER,
    price REAL
)
""")

# 2. Insert sample data
cursor.executemany("""
INSERT INTO housing VALUES (?, ?, ?, ?)
""", [
    (50000, 10, 3, 200000),
    (60000, 5, 4, 250000),
    (45000, 20, 2, 150000)
])

# Save changes
conn.commit()

# 3. Retrieve all records
print("=== All Housing Records ===")
cursor.execute("SELECT * FROM housing")
print(cursor.fetchall())

# 4. Filter records
print("\n=== Properties with Price > 180,000 ===")
cursor.execute("SELECT * FROM housing WHERE price > 180000")
print(cursor.fetchall())

# Clean up connection
conn.close()


print("\n📌 Selected Columns:")
cursor.execute("SELECT income, price FROM housing")
print(cursor.fetchall())



print("\n📊 Average Price:")
cursor.execute("SELECT AVG(price) FROM housing")
print(cursor.fetchone())

print("\n💰 Total Income:")
cursor.execute("SELECT SUM(income) FROM housing")
print(cursor.fetchone())

print("\n🔢 Total Rows:")
cursor.execute("SELECT COUNT(*) FROM housing")
print(cursor.fetchone())




print("\n🏠 Avg Price per Room Count:")
cursor.execute("""
SELECT rooms, AVG(price)
FROM housing
GROUP BY rooms
""")
print(cursor.fetchall())