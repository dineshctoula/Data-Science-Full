#!/usr/bin/env python3
"""
100-Day Data Science Challenge
Day 2: Basic Data Types & Arithmetic Operations

This script serves as a study and coding exercise guide for Day 2.
We cover:
1. Integer and Float arithmetic operations (+, -, *, /, //, %, **).
2. String operations (indexing, slicing, standard methods, f-strings).
3. Booleans (comparison operators, logical operators).
4. Practical Exercise: An interactive bill splitter and tip calculator.
"""

import sys

def run_arithmetic_demos():
    """Demonstrates basic arithmetic operations in Python with detailed comments."""
    print("\n--- 1. Numeric Types & Arithmetic Operations ---")
    
    # 1. Variable Assignment (Python dynamically infers the type)
    x = 17  # An Integer (int)
    y = 5   # An Integer (int)
    
    print(f"Variables: x = {x} (type: {type(x).__name__}), y = {y} (type: {type(y).__name__})")
    
    # 2. Basic Arithmetic Operators
    print(f"Addition (x + y):        {x + y}")        # Sum of x and y
    print(f"Subtraction (x - y):     {x - y}")        # Difference of x and y
    print(f"Multiplication (x * y):  {x * y}")        # Product of x and y
    
    # 3. Division Operators (Crucial for interviews!)
    # Standard division (/) always returns a float in Python 3, even if it divides evenly.
    print(f"Standard Division (x/y): {x / y} (type: {type(x / y).__name__})")
    
    # Floor division (//) divides and truncates the fractional part, returning the mathematical floor.
    print(f"Floor Division (x // y): {x // y} (rounds down to nearest whole number)")
    
    # Modulo (%) returns the remainder of the integer division. Useful for finding even/odd or cycles.
    print(f"Modulo (x % y):          {x % y} (remainder of {x} divided by {y})")
    
    # Exponentiation (**) raises the base to the power of the exponent.
    print(f"Exponentiation (x ** y): {x ** y} ({x} raised to the power of {y})")
    
    # 4. Floating Point Precision Quirks
    # Computer systems represent floats in base-2 (binary) fractions. This causes small rounding issues
    # that are classic programming and data science interview questions (e.g., 0.1 + 0.2 != 0.3).
    f1 = 0.1
    f2 = 0.2
    print(f"\nFloating Point precision example:")
    print(f"Raw 0.1 + 0.2 = {f1 + f2} (due to binary floating-point representation)")
    print(f"Corrected with round(): {round(f1 + f2, 2)} (rounded to 2 decimal places)")


def run_string_demos():
    """Demonstrates string manipulation, slicing, and methods with comments."""
    print("\n--- 2. String Manipulation & Formatting ---")
    
    # 1. String Slicing: string[start:stop:step]
    # - start: starting index (inclusive, defaults to 0)
    # - stop: ending index (exclusive, defaults to len(string))
    # - step: step increment (defaults to 1; negative step reverses the string)
    text = "Data Science Challenge"
    print(f"Original Text: '{text}'")
    print(f"Length of string (len(text)): {len(text)}")
    print(f"First 4 characters (text[:4]):       '{text[:4]}'")       # Indexes 0, 1, 2, 3
    print(f"Last 9 characters (text[-9:]):       '{text[-9:]}'")      # Negative indexing starts from the right
    print(f"Slice with step 2 (text[::2]):       '{text[::2]}'")      # Extracts every second character
    print(f"Reversed string (text[::-1]):        '{text[::-1]}'")     # Step of -1 reverses sequence
    
    # 2. Common String Methods (Strings are immutable, so these return NEW strings)
    messy_text = "  python programming is awesome!   "
    print(f"\nMessy text: '{messy_text}'")
    
    # strip() removes leading and trailing whitespace.
    print(f"Stripped:                            '{messy_text.strip()}'")
    # upper() converts all characters to uppercase.
    print(f"Uppercase:                           '{messy_text.strip().upper()}'")
    # title() converts the first letter of each word to uppercase.
    print(f"Title Case:                          '{messy_text.strip().title()}'")
    # replace() replaces a substring with another.
    print(f"Replace 'awesome' with 'essential':  '{messy_text.strip().replace('awesome', 'essential')}'")
    
    # 3. Split & Join
    # split() splits the string on spaces and returns a list of strings.
    words = messy_text.strip().split()
    print(f"Split words into a list:             {words}")
    # join() merges a list of strings using the caller string as a separator.
    print(f"Joined with hyphens:                 '{'-'.join(words)}'")
    
    # 4. f-strings (Formated String Literals - Introduced in Python 3.6, clean and efficient)
    name = "Dinesh"
    progress = 2
    # :02d format specifier formats integers with leading zeros to be at least 2 digits wide.
    message = f"Hello {name}, welcome to Day {progress:02d}!"
    print(f"\nf-string formatting: '{message}'")


def run_boolean_demos():
    """Demonstrates boolean values, comparison, and logical operators with comments."""
    print("\n--- 3. Booleans & Logical Operators ---")
    
    a = 10
    b = 20
    c = 10
    
    print(f"Variables: a = {a}, b = {b}, c = {c}")
    
    # 1. Comparison Operators (return True or False)
    print(f"Is a equal to c? (a == c):          {a == c}")
    print(f"Is a not equal to b? (a != b):      {a != b}")
    print(f"Is a greater than b? (a > b):       {a > b}")
    print(f"Is a less than or equal to c? (a <= c): {a <= c}")
    
    # 2. Logical Operators (and, or, not)
    # 'and' requires BOTH conditions to be True.
    print(f"\nLogical operations:")
    print(f"(a == c) and (b > a):               {(a == c) and (b > a)} (both are True)")
    # 'or' requires AT LEAST ONE condition to be True.
    print(f"(a == b) or (b > a):                {(a == b) or (b > a)} (one is True)")
    # 'not' reverses the boolean value.
    print(f"not (a == b):                       {not (a == b)} (reverses False to True)")


def get_float_input(prompt: str) -> float:
    """Prompts for float input and validates that it is a positive number."""
    while True:
        try:
            val_str = input(prompt).strip()
            val = float(val_str)
            if val < 0:
                print("Error: Value cannot be negative. Please try again.")
                continue
            return val
        except ValueError:
            print("Error: Please enter a valid number (integer or decimal).")
        except (KeyboardInterrupt, EOFError):
            print("\nCalculator exit.")
            sys.exit(0)


def get_int_input(prompt: str) -> int:
    """Prompts for integer input and validates that it is at least 1."""
    while True:
        try:
            val_str = input(prompt).strip()
            val = int(val_str)
            if val <= 0:
                print("Error: Value must be at least 1 (cannot split among 0 or negative people).")
                continue
            return val
        except ValueError:
            print("Error: Please enter a valid whole number.")
        except (KeyboardInterrupt, EOFError):
            print("\nCalculator exit.")
            sys.exit(0)


def run_interactive_calculator():
    """Interactive bill splitter and tip calculator using basic data types."""
    print("\n==============================================")
    print("💸 Day 2 Exercise: Tip & Bill Splitter 💸")
    print("==============================================")
    
    # 1. Get validated user inputs
    total_bill = get_float_input("Enter the total bill amount ($): ")
    tip_percentage = get_float_input("Enter the tip percentage to give (e.g., 15 for 15%): ")
    number_of_people = get_int_input("Enter the number of people to split the bill: ")
    
    # 2. Perform arithmetic operations
    tip_amount = total_bill * (tip_percentage / 100)
    grand_total = total_bill + tip_amount
    amount_per_person = grand_total / number_of_people
    
    # 3. Print receipt summary with clean f-string alignments
    # :10.2f formats the float to be right-aligned with 10 total character spaces and 2 decimal places.
    print("\n---------------- Receipt Summary ----------------")
    print(f"Subtotal:         ${total_bill:10.2f}")
    print(f"Tip ({tip_percentage:.1f}%):     ${tip_amount:10.2f}")
    print(f"Grand Total:      ${grand_total:10.2f}")
    print(f"Split (among {number_of_people:d}): ${amount_per_person:10.2f} each")
    print("-------------------------------------------------")
    print("Day 2 Exercise completed successfully!\n")


def main():
    print("==================================================")
    print("🚀 Day 2: Basic Data Types & Operations 🚀")
    print("==================================================")
    
    # Run structured concept demonstrations
    run_arithmetic_demos()
    run_string_demos()
    run_boolean_demos()
    
    # Run the interactive bill splitter exercise
    run_interactive_calculator()

if __name__ == "__main__":
    main()
