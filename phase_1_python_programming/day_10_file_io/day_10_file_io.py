#!/usr/bin/env python3
"""
100-Day Data Science Challenge
Day 10: File Input/Output (I/O) in Python

Topics Covered:
1. Writing and Appending to Text Files (modes 'w', 'a')
2. Reading Text Files (read, readline, readlines, context managers)
3. Exception Handling during File Operations (FileNotFoundError)
4. Working with JSON files (json.dump, json.load)
5. Practical Exercise: Processing a CSV dataset and generating a summary report
"""

import os
import json
import csv

def demonstrate_writing_files():
    print("=== 1. WRITING & APPENDING TO FILES ===")
    
    # 1. Writing to a new file (overwrites existing content)
    with open("students.txt", "w") as file:
        file.write("Alice,85\n")
        file.write("Bob,90\n")
        print("Created students.txt and wrote initial data.")
        
    # 2. Appending to an existing file
    with open("students.txt", "a") as file:
        file.write("Charlie,95\n")
        print("Appended Charlie's data to students.txt.")
    print()


def demonstrate_reading_files():
    print("=== 2. READING FILES ===")
    
    # 1. Reading the entire file content at once
    print("--- Reading entire file using read() ---")
    with open("students.txt", "r") as file:
        content = file.read()
        print(content.strip())
        
    # 2. Reading line by line using readline()
    print("--- Reading line by line using readline() ---")
    with open("students.txt", "r") as file:
        line1 = file.readline().strip()
        line2 = file.readline().strip()
        print(f"Line 1: {line1}")
        print(f"Line 2: {line2}")
        
    # 3. Reading all lines into a list using readlines()
    print("--- Reading all lines into a list using readlines() ---")
    with open("students.txt", "r") as file:
        lines = file.readlines()
        print(f"Lines list: {[line.strip() for line in lines]}")
        
    # 4. Iterating over the file object directly (memory efficient & best practice)
    print("--- Iterating directly over the file object ---")
    with open("students.txt", "r") as file:
        for idx, line in enumerate(file, start=1):
            print(f"Row {idx}: {line.strip()}")
    print()


def demonstrate_file_exceptions():
    print("=== 3. FILE EXCEPTION HANDLING ===")
    
    # Handle FileNotFoundError gracefully
    non_existent_file = "missing_dataset.csv"
    try:
        with open(non_existent_file, "r") as file:
            data = file.read()
            print(data)
    except FileNotFoundError:
        print(f"Error: The file '{non_existent_file}' could not be found.")
    print()


def demonstrate_json_operations():
    print("=== 4. JSON FILE OPERATIONS ===")
    
    config_data = {
        "model": "Random Forest",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10
        },
        "accuracy": 0.89
    }
    
    # 1. Writing JSON data to a file
    with open("config.json", "w") as json_file:
        json.dump(config_data, json_file, indent=4)
        print("Configuration saved to config.json.")
        
    # 2. Reading JSON data from a file
    with open("config.json", "r") as json_file:
        loaded_data = json.load(json_file)
        print("Loaded JSON Config:")
        print(f"  Model: {loaded_data['model']}")
        print(f"  Accuracy: {loaded_data['accuracy']}")
    print()


def run_practical_exercise():
    print("=== 5. PRACTICAL DATA SCIENCE EXERCISE ===")
    
    # 1. Create a raw CSV dataset of sales transactions
    sales_data = [
        ["transaction_id", "product", "price", "quantity"],
        ["T1001", "Laptop", "1200", "1"],
        ["T1002", "Mouse", "25", "2"],
        ["T1003", "Keyboard", "invalid_price", "1"],  # Corrupt entry
        ["T1004", "Monitor", "300", "2"],
        ["T1005", "Cable", "15", "5"]
    ]
    
    with open("raw_sales.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(sales_data)
    print("Created raw_sales.csv for exercise.")
    
    # 2. Process the CSV file: clean/parse the numeric values and handle errors
    total_revenue = 0.0
    valid_transactions = 0
    corrupt_records = 0
    
    with open("raw_sales.csv", "r") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)  # Skip header
        
        for row in reader:
            tx_id, product, price_str, qty_str = row
            try:
                price = float(price_str)
                qty = int(qty_str)
                revenue = price * qty
                total_revenue += revenue
                valid_transactions += 1
            except ValueError:
                print(f"  [Warning] Skipped corrupt row: {row}")
                corrupt_records += 1
                
    # 3. Generate a summary report text file
    report_content = (
        "=== SALES DATA PROCESSING REPORT ===\n"
        f"Total Transactions Processed: {valid_transactions + corrupt_records}\n"
        f"Successful Transactions: {valid_transactions}\n"
        f"Corrupted/Skipped Records: {corrupt_records}\n"
        f"Total Revenue Generated: ${total_revenue:.2f}\n"
    )
    
    with open("sales_report.txt", "w") as report_file:
        report_file.write(report_content)
    print("Generated sales_report.txt successfully.")
    
    # Print the report content to the console
    print("\nReport Output:")
    print(report_content.strip())
    
    # Clean up files created during demo
    for temp_file in ["students.txt", "config.json", "raw_sales.csv", "sales_report.txt"]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"Cleaned up temporary file: {temp_file}")
            
    print("\nDay 10 File I/O exercise completed successfully!")


if __name__ == "__main__":
    demonstrate_writing_files()
    demonstrate_reading_files()
    demonstrate_file_exceptions()
    demonstrate_json_operations()
    run_practical_exercise()
