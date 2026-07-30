import sqlite3

conn = sqlite3.connect("housing.db")
cursor = conn.cursor()