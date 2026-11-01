#!/usr/bin/env python3
"""
====================================================
100 Days of Data Science - Day 15
Topic: NumPy Mathematical & Statistical Functions

Topics Covered
--------------
1. Basic Statistics (mean, median, std, var, sum, min, max)
2. Axis-wise Operations (row-wise vs column-wise)
3. Searching & Locating (argmin, argmax, np.where)
4. Practical Exercise (Student Exam Grades Analysis)

Author: Dinesh Sitoula
====================================================
"""

import numpy as np

# -----------------------------------
# 1. BASIC STATISTICS
# -----------------------------------
def basic_statistics():
    print("=" * 60)
    print("1. BASIC STATISTICAL METRICS")
    print("=" * 60)

    # Creating a sample dataset of house prices (in thousands)
    prices = np.array([250, 310, 420, 280, 560, 390, 850, 290])
    print("Prices dataset (1D Array):", prices)
    
    print("\nCalculations:")
    print("Total Sum :", np.sum(prices))
    print("Mean (Average) :", np.mean(prices))
    print("Median :", np.median(prices))
    print("Minimum :", np.min(prices))
    print("Maximum :", np.max(prices))
    print("Standard Deviation :", np.round(np.std(prices), 2))
    print("Variance :", np.round(np.var(prices), 2))


# -----------------------------------
# 2. AXIS-WISE OPERATIONS
# -----------------------------------
def axis_operations():
    print("\n" + "=" * 60)
    print("2. AXIS-WISE OPERATIONS ON 2D ARRAYS")
    print("=" * 60)

    # 3x4 Matrix representing 3 users' scores across 4 rounds
    scores = np.array([
        [10, 15, 20, 18],
        [22, 28, 25, 30],
        [15, 12, 18, 14]
    ])
    print("Scores Matrix (3x4):\n", scores)

    # Column-wise statistics (axis=0)
    # Calculates summary stats for each column (round)
    mean_per_round = np.mean(scores, axis=0)
    print("\nColumn-wise Mean (axis=0 - Average per Round):")
    print(mean_per_round)

    # Row-wise statistics (axis=1)
    # Calculates summary stats for each row (user)
    total_per_user = np.sum(scores, axis=1)
    print("\nRow-wise Sum (axis=1 - Total Score per User):")
    print(total_per_user)


# -----------------------------------
# 3. SEARCHING & LOCATING (ARGMIN, ARGMAX, WHERE)
# -----------------------------------
def search_and_locate():
    print("\n" + "=" * 60)
    print("3. SEARCHING & LOCATING ELEMENTS")
    print("=" * 60)

    temperatures = np.array([18.5, 22.1, 15.4, 29.8, 12.3, 24.5])
    print("Temperatures:", temperatures)

    # argmin & argmax return the index of the min/max element
    min_idx = np.argmin(temperatures)
    max_idx = np.argmax(temperatures)
    print(f"Coldest Temperature Index: {min_idx} (Value: {temperatures[min_idx]})")
    print(f"Hottest Temperature Index: {max_idx} (Value: {temperatures[max_idx]})")

    # np.where returns indices where conditions are met
    hot_days_indices = np.where(temperatures > 22.0)
    print("\nDays with Temperature > 22.0 (indices):", hot_days_indices[0])
    print("Temperature values for those days:", temperatures[hot_days_indices])

    # np.where with conditional replacements: np.where(condition, x, y)
    # Replaces values with 'Hot' or 'Cold'
    climate_labels = np.where(temperatures > 20.0, "Warm", "Cool")
    print("\nClimate Classification:", climate_labels)


# -----------------------------------
# 4. PRACTICAL EXERCISE
# -----------------------------------
def practical_exercise():
    print("\n" + "=" * 60)
    print("4. PRACTICAL EXERCISE: EXAM GRADES ANALYSIS")
    print("=" * 60)
    print("Task: Create a 4x4 matrix representing exam grades (students as rows, subjects as columns).")
    print("      Find the average score for each subject, and the top-performing student index.")

    # 1. Create a 4x4 matrix of grades
    # Rows: Student A, Student B, Student C, Student D
    # Columns: Math, Science, English, History
    grades = np.array([
        [85, 90, 78, 92],  # Student 0
        [72, 85, 88, 80],  # Student 1
        [95, 92, 96, 98],  # Student 2 (Topper)
        [68, 70, 75, 72]   # Student 3
    ])

    subjects = ["Math", "Science", "English", "History"]
    students = ["Student A", "Student B", "Student C", "Student D"]

    print("\nGrades Matrix (4x4):")
    print(grades)

    # 2. Average score for each subject (Column-wise mean)
    subject_averages = np.mean(grades, axis=0)
    print("\nAverage score per subject (Column-wise Mean):")
    for i, sub in enumerate(subjects):
        print(f"  {sub}: {subject_averages[i]:.2f}")

    # 3. Top-performing student (Row-wise averages first, then argmax)
    student_averages = np.mean(grades, axis=1)
    topper_idx = np.argmax(student_averages)

    print("\nStudent Averages:")
    for i, stud in enumerate(students):
        print(f"  {stud}: {student_averages[i]:.2f}")

    print(f"\nTop-performing Student: {students[topper_idx]} (Index: {topper_idx})")
    print(f"Highest Average Grade: {student_averages[topper_idx]:.2f}%")


def main():
    print("=== Day 15: NumPy Mathematical & Statistical Functions ===")
    basic_statistics()
    axis_operations()
    search_and_locate()
    practical_exercise()
    print("\nDay 15 Completed Successfully!")

if __name__ == "__main__":
    main()
