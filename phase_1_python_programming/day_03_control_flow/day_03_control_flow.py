#!/usr/bin/env python3
"""
100-Day Data Science Challenge
Day 3: Control Flow & Conditional Logic

Topics Covered:
1. if, elif, else statements
2. Logical operators (and, or, not)
3. Nested conditions and ternary operators
4. Practical Exercise: Grade Calculator (0-100 to A/B/C/D/F)
"""

def demonstrate_conditionals():
    print("\n--- 1. Conditional Statements & Logical Operators ---")
    # Example 1: Simple check
    age = 20
    is_student = True
    
    # Using 'if', 'elif', 'else' along with logical operators 'and', 'or', 'not'
    if age >= 18 and is_student:
        print("Discounted student ticket: $10")
    elif age >= 18:
        print("Adult ticket: $15")
    else:
        print("Child ticket: $5")

    # Example 2: Ternary Operator (One-liner if-else)
    # syntax: value_if_true if condition else value_if_false
    status = "Adult" if age >= 18 else "Minor"
    print(f"Ternary operator example: Age {age} is categorized as {status}")


def calculate_grade(score):
    """
    Exercise: Grade Calculator
    Takes a numerical score (0-100) and returns the corresponding letter grade.
    
    Grading boundaries:
    - 90 - 100: A
    - 80 - 89: B
    - 70 - 79: C
    - 60 - 69: D
    - Below 60: F
    
    Must handle invalid input (scores less than 0 or greater than 100).
    """
    # TODO: Implement the grade calculation logic using if, elif, else
    pass


def main():
    print("=== Day 3: Control Flow & Conditional Logic ===")
    
    # Run the demo
    demonstrate_conditionals()
    
    # Interactive test of the Grade Calculator
    print("\n--- 2. Practical Exercise: Grade Calculator ---")
    try:
        user_input = input("Enter a numerical score (0-100): ").strip()
        score = float(user_input)
        
        grade = calculate_grade(score)
        print(f"The letter grade for score {score} is: {grade}")
    except ValueError:
        print("Invalid input! Please enter a valid number.")

if __name__ == "__main__":
    main()
