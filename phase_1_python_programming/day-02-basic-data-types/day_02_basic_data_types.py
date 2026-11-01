```python
#!/usr/bin/env python3
"""
==================================================
100-Day Data Science Challenge
Day 2: Basic Data Types & Arithmetic Operations
==================================================

Topics Covered:
1. Integer and Float Arithmetic
2. String Operations and Formatting
3. Boolean and Logical Operators
4. Practical Exercise: Bill Splitter & Tip Calculator

Author: Dinesh Sitoula
"""

import sys


# --------------------------------------------------
# 1. NUMERIC TYPES & ARITHMETIC OPERATIONS
# --------------------------------------------------
def run_arithmetic_demos():
    """Demonstrate basic arithmetic operations in Python."""

    print("\n--- 1. Numeric Types & Arithmetic Operations ---")

    x = 17
    y = 5

    print(
        f"Variables: x = {x} "
        f"(type: {type(x).__name__}), "
        f"y = {y} "
        f"(type: {type(y).__name__})"
    )

    print(f"Addition (x + y):        {x + y}")
    print(f"Subtraction (x - y):     {x - y}")
    print(f"Multiplication (x * y):  {x * y}")
    print(f"Standard Division (x / y): {x / y}")
    print(f"Floor Division (x // y):   {x // y}")
    print(f"Modulo (x % y):            {x % y}")
    print(f"Exponentiation (x ** y):   {x ** y}")

    # Floating-point precision example
    f1 = 0.1
    f2 = 0.2

    print("\nFloating-Point Precision Example:")
    print(f"0.1 + 0.2 = {f1 + f2}")
    print(f"Rounded Result = {round(f1 + f2, 2)}")


# --------------------------------------------------
# 2. STRING OPERATIONS & FORMATTING
# --------------------------------------------------
def run_string_demos():
    """Demonstrate string indexing, slicing, methods, and formatting."""

    print("\n--- 2. String Manipulation & Formatting ---")

    text = "Data Science Challenge"

    print(f"Original Text: '{text}'")
    print(f"Length: {len(text)}")
    print(f"First 4 Characters: '{text[:4]}'")
    print(f"Last 9 Characters: '{text[-9:]}'")
    print(f"Every Second Character: '{text[::2]}'")
    print(f"Reversed String: '{text[::-1]}'")

    # Common string methods
    messy_text = "  python programming is awesome!   "

    print(f"\nOriginal: '{messy_text}'")
    print(f"Stripped: '{messy_text.strip()}'")
    print(f"Uppercase: '{messy_text.strip().upper()}'")
    print(f"Title Case: '{messy_text.strip().title()}'")

    replaced_text = messy_text.strip().replace(
        "awesome",
        "essential"
    )

    print(f"Replaced: '{replaced_text}'")

    # Split and join
    words = messy_text.strip().split()

    print(f"Split Words: {words}")
    print(f"Joined Words: {'-'.join(words)}")

    # f-string formatting
    name = "Dinesh"
    progress = 2

    message = f"Hello {name}, welcome to Day {progress:02d}!"

    print(f"\nf-string Example: '{message}'")


# --------------------------------------------------
# 3. BOOLEAN & LOGICAL OPERATORS
# --------------------------------------------------
def run_boolean_demos():
    """Demonstrate comparison and logical operators."""

    print("\n--- 3. Booleans & Logical Operators ---")

    a = 10
    b = 20
    c = 10

    print(f"Variables: a = {a}, b = {b}, c = {c}")

    # Comparison operators
    print(f"\na == c: {a == c}")
    print(f"a != b: {a != b}")
    print(f"a > b:  {a > b}")
    print(f"a <= c: {a <= c}")

    # Logical operators
    print("\nLogical Operations:")

    print(
        f"(a == c) and (b > a): "
        f"{(a == c) and (b > a)}"
    )

    print(
        f"(a == b) or (b > a): "
        f"{(a == b) or (b > a)}"
    )

    print(f"not (a == b): {not (a == b)}")


# --------------------------------------------------
# 4. INPUT VALIDATION
# --------------------------------------------------
def get_float_input(prompt: str) -> float:
    """Get a non-negative floating-point number from the user."""

    while True:
        try:
            value = float(input(prompt).strip())

            if value < 0:
                print("Error: Value cannot be negative.")
                continue

            return value

        except ValueError:
            print("Error: Please enter a valid number.")

        except (KeyboardInterrupt, EOFError):
            print("\nCalculator exited.")
            sys.exit(0)


def get_int_input(prompt: str) -> int:
    """Get a positive integer from the user."""

    while True:
        try:
            value = int(input(prompt).strip())

            if value <= 0:
                print("Error: Number of people must be at least 1.")
                continue

            return value

        except ValueError:
            print("Error: Please enter a valid whole number.")

        except (KeyboardInterrupt, EOFError):
            print("\nCalculator exited.")
            sys.exit(0)


# --------------------------------------------------
# 5. BILL SPLITTER & TIP CALCULATOR
# --------------------------------------------------
def run_interactive_calculator():
    """Calculate tip, total bill, and amount per person."""

    print("\n" + "=" * 46)
    print("💸 Day 2 Exercise: Tip & Bill Splitter")
    print("=" * 46)

    total_bill = get_float_input(
        "Enter the total bill amount ($): "
    )

    tip_percentage = get_float_input(
        "Enter the tip percentage (e.g., 15): "
    )

    number_of_people = get_int_input(
        "Enter the number of people: "
    )

    # Calculate tip and final bill
    tip_amount = total_bill * (tip_percentage / 100)
    grand_total = total_bill + tip_amount
    amount_per_person = grand_total / number_of_people

    # Display receipt
    print("\n---------------- Receipt Summary ----------------")

    print(f"Subtotal:       ${total_bill:10.2f}")
    print(f"Tip ({tip_percentage:.1f}%):    ${tip_amount:10.2f}")
    print(f"Grand Total:    ${grand_total:10.2f}")
    print(
        f"Per Person ({number_of_people}): "
        f"${amount_per_person:10.2f}"
    )

    print("-------------------------------------------------")
    print("Day 2 Exercise completed successfully!")


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------
def main():
    """Run all Day 2 demonstrations and exercises."""

    print("=" * 50)
    print("🚀 Day 2: Basic Data Types & Operations 🚀")
    print("=" * 50)

    run_arithmetic_demos()
    run_string_demos()
    run_boolean_demos()
    run_interactive_calculator()


# --------------------------------------------------
# PROGRAM ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    main()
```
