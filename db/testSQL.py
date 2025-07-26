# create_huge_test_db.py

import sqlite3
from faker import Faker
import random

# Initialize
fake = Faker()
conn = sqlite3.connect("huge_test.db")
cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    age INTEGER,
    country TEXT,
    signup_date TEXT,
    is_active INTEGER
);
""")

# Insert 10,000 fake users
BATCH_SIZE = 1000
TOTAL_RECORDS = 10000

for batch_start in range(0, TOTAL_RECORDS, BATCH_SIZE):
    users = []
    for _ in range(BATCH_SIZE):
        users.append((
            fake.name(),
            fake.email(),
            random.randint(18, 70),
            fake.country(),
            fake.date_between(start_date='-5y', end_date='today').isoformat(),
            random.choice([0, 1])
        ))
    
    cursor.executemany("""
    INSERT INTO users (name, email, age, country, signup_date, is_active)
    VALUES (?, ?, ?, ?, ?, ?)
    """, users)
    
    print(f"✅ Inserted records {batch_start + 1} to {batch_start + BATCH_SIZE}")

conn.commit()
conn.close()
print("🎉 huge_test.db created with 10,000 users!")
