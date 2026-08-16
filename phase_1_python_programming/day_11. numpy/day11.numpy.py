```python
#!/usr/bin/env python3
"""
====================================================
100 Days of Data Science - Day 11
Topic: NumPy Fundamentals

Commit 1:
Refactor core NumPy demonstrations.

Topics Covered
--------------
1. Creating Arrays
2. Array Attributes
3. Indexing and Slicing
4. Mathematical Operations
5. Statistical Functions
6. Reshaping Arrays
7. Random Numbers
8. Student Marks Analysis

Author: Dinesh Sitoula
====================================================
"""

import numpy as np


def print_section(title):
    """Display a formatted section heading."""
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


# ==================================================
# 1. ARRAY CREATION
# ==================================================

def demonstrate_array_creation():
    """Demonstrate different ways to create NumPy arrays."""

    print_section("1. ARRAY CREATION")

    array = np.array([10, 20, 30, 40, 50])
    zeros = np.zeros((2, 3))
    ones = np.ones((3, 3))
    range_array = np.arange(1, 11)
    linear_array = np.linspace(0, 100, 5)

    print("Array:", array)
    print("\nZeros:\n", zeros)
    print("\nOnes:\n", ones)
    print("\nRange:", range_array)
    print("\nLinspace:", linear_array)


# ==================================================
# 2. ARRAY ATTRIBUTES
# ==================================================

def demonstrate_array_attributes():
    """Demonstrate important NumPy array attributes."""

    print_section("2. ARRAY ATTRIBUTES")

    array = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    print("Array:\n", array)
    print("\nShape:", array.shape)
    print("Dimensions:", array.ndim)
    print("Size:", array.size)
    print("Data Type:", array.dtype)


# ==================================================
# 3. INDEXING AND SLICING
# ==================================================

def demonstrate_indexing_and_slicing():
    """Demonstrate NumPy indexing and slicing."""

    print_section("3. INDEXING & SLICING")

    array = np.array([5, 10, 15, 20, 25, 30])

    print("Array:", array)
    print("First Element:", array[0])
    print("Last Element:", array[-1])
    print("Slice:", array[1:5])


# ==================================================
# 4. MATHEMATICAL OPERATIONS
# ==================================================

def demonstrate_mathematical_operations():
    """Demonstrate element-wise mathematical operations."""

    print_section("4. MATHEMATICAL OPERATIONS")

    first_array = np.array([10, 20, 30])
    second_array = np.array([2, 4, 5])

    print("Array A:", first_array)
    print("Array B:", second_array)

    print("\nAddition:", first_array + second_array)
    print("Subtraction:", first_array - second_array)
    print("Multiplication:", first_array * second_array)
    print("Division:", first_array / second_array)


# ==================================================
# 5. STATISTICAL FUNCTIONS
# ==================================================

def demonstrate_statistics():
    """Demonstrate basic statistical functions."""

    print_section("5. STATISTICAL FUNCTIONS")

    marks = np.array([78, 82, 91, 65, 88, 95])

    print("Marks:", marks)
    print("\nMean:", np.mean(marks))
    print("Median:", np.median(marks))
    print("Maximum:", np.max(marks))
    print("Minimum:", np.min(marks))
    print("Standard Deviation:", np.std(marks))


# ==================================================
# 6. RESHAPING ARRAYS
# ==================================================

def demonstrate_reshape():
    """Demonstrate reshaping a NumPy array."""

    print_section("6. RESHAPING ARRAYS")

    array = np.arange(1, 13)
    matrix = array.reshape(3, 4)

    print("Original Array:", array)
    print("\nReshaped Array:\n", matrix)


# ==================================================
# 7. RANDOM NUMBERS
# ==================================================

def demonstrate_random_numbers():
    """Generate random student marks."""

    print_section("7. RANDOM NUMBERS")

    random_marks = np.random.randint(40, 101, size=10)

    print("Random Marks:", random_marks)


# ==================================================
# MAIN FUNCTION
# ==================================================

def main():
    """Run all NumPy demonstrations."""

    demonstrate_array_creation()
    demonstrate_array_attributes()
    demonstrate_indexing_and_slicing()
    demonstrate_mathematical_operations()
    demonstrate_statistics()
    demonstrate_reshape()
    demonstrate_random_numbers()

    print("\n" + "=" * 50)
    print("Core NumPy Demonstrations Completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
```



```python
#!/usr/bin/env python3
"""
====================================================
100 Days of Data Science - Day 11
Topic: NumPy Fundamentals

Commit 2:
Add practical student marks analysis using NumPy.

Author: Dinesh Sitoula
====================================================
"""

import numpy as np


def print_section(title):
    """Display a formatted section heading."""
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def analyze_student_marks():
    """Analyze student marks using NumPy."""

    print_section("8. MINI PROJECT - STUDENT MARKS ANALYSIS")

    students = np.array([
        [78, 80, 75],
        [92, 88, 95],
        [65, 70, 60],
        [85, 90, 84],
        [72, 75, 70]
    ])

    # Calculate average marks for each student.
    student_averages = np.mean(students, axis=1)

    # Find the student with the highest average.
    topper_index = np.argmax(student_averages)
    topper_average = student_averages[topper_index]

    # Calculate average marks for each subject.
    subject_averages = np.mean(students, axis=0)

    print("\nStudent Marks:")
    print(students)

    print("\nAverage Marks of Each Student:")
    print(student_averages)

    print("\nTopper Student Index:", topper_index)
    print("Highest Average:", topper_average)

    print("\nSubject-Wise Average:")
    print(subject_averages)


def main():
    """Run the student marks analysis."""

    analyze_student_marks()

    print("\n" + "=" * 50)
    print("Student Analysis Completed Successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
```

