"""
============================================================
DAY 9 - EXCEPTION HANDLING IN PYTHON
100 DAYS OF DATA SCIENCE
============================================================

Exception handling allows a program to handle runtime errors
without crashing unexpectedly.

Main concepts covered:
    1. Basic try-except
    2. Multiple exceptions
    3. Exception objects
    4. else block
    5. finally block
    6. raise statement
    7. Custom exceptions
    8. Nested exception handling
    9. Exception handling in functions
    10. Data cleaning using exceptions
    11. CSV-style data cleaning
    12. Handling multiple exceptions together

Basic syntax:

try:
    # Risky code
except:
    # Handle exception
else:
    # Executes when no exception occurs
finally:
    # Always executes
"""


def basic_try_except():
    """Demonstrate basic try-except handling."""
    print("\n1. BASIC TRY-EXCEPT")

    try:
        number = int(input("Enter a number: "))
        print(f"You entered: {number}")

    except ValueError:
        print("Invalid input! Please enter a valid integer.")


def multiple_exceptions():
    """Handle different types of exceptions separately."""
    print("\n2. MULTIPLE EXCEPTIONS")

    try:
        numerator = int(input("Enter numerator: "))
        denominator = int(input("Enter denominator: "))

        result = numerator / denominator
        print("Result:", result)

    except ValueError:
        print("Please enter valid integer numbers.")

    except ZeroDivisionError:
        print("Cannot divide by zero.")


def exception_object():
    """Demonstrate how to access an exception object."""
    print("\n3. EXCEPTION OBJECT")

    try:
        number = int(input("Enter a number: "))
        result = 100 / number

        print("Result:", result)

    except (ValueError, ZeroDivisionError) as error:
        print("Error occurred:", error)


def else_block():
    """Demonstrate the else block."""
    print("\n4. ELSE BLOCK")

    try:
        age = int(input("Enter your age: "))

    except ValueError:
        print("Age must be a valid number.")

    else:
        print("Input successful.")
        print("Your age is:", age)


def finally_block():
    """Demonstrate the finally block with safe file handling."""
    print("\n5. FINALLY BLOCK")

    try:
        with open("sample.txt", "r", encoding="utf-8") as file:
            content = file.read()

        print("File content:")
        print(content)

    except FileNotFoundError:
        print("File not found.")

    finally:
        print("File operation completed.")


def raise_exception():
    """Demonstrate raising a built-in exception."""
    print("\n6. RAISE EXCEPTION")

    try:
        salary = float(input("Enter salary: "))

        if salary < 0:
            raise ValueError("Salary cannot be negative.")

        print(f"Salary: ${salary:,.2f}")

    except ValueError as error:
        print("Error:", error)


class InvalidAgeError(Exception):
    """Custom exception for invalid age."""

    pass


def custom_exception():
    """Demonstrate a user-defined exception."""
    print("\n7. CUSTOM EXCEPTION")

    try:
        age = int(input("Enter age: "))

        if age < 18:
            raise InvalidAgeError("Age must be 18 or above.")

        print("Eligible.")

    except ValueError:
        print("Please enter a valid age.")

    except InvalidAgeError as error:
        print("Invalid age:", error)


def nested_exception():
    """Demonstrate nested try-except blocks."""
    print("\n8. NESTED TRY-EXCEPT")

    try:
        try:
            number = int(input("Enter a number: "))
            result = 50 / number
            print("Result:", result)

        except ZeroDivisionError:
            print("Inner: Division by zero.")

    except ValueError:
        print("Outer: Invalid input.")


def divide_numbers(first_number, second_number):
    """
    Divide two numbers safely.

    Returns:
        float: Division result.
        str: Error message if division by zero occurs.
    """
    try:
        return first_number / second_number

    except ZeroDivisionError:
        return "Cannot divide by zero."


def function_example():
    """Demonstrate exception handling inside a function."""
    print("\n9. FUNCTION EXAMPLE")

    print("10 / 2 =", divide_numbers(10, 2))
    print("10 / 0 =", divide_numbers(10, 0))


def data_science_example():
    """Demonstrate exception handling during data cleaning."""
    print("\n10. DATA SCIENCE EXAMPLE")

    dataset = ["100", "200", "abc", "400", "xyz"]

    clean_data = []

    for value in dataset:
        try:
            clean_data.append(int(value))

        except ValueError:
            print(f"Invalid data skipped: {value}")

    print("Clean Data:", clean_data)


def csv_data_cleaning():
    """Simulate cleaning age values from a CSV dataset."""
    print("\n11. CSV DATA CLEANING")

    raw_data = [
        "25",
        "30",
        "unknown",
        "45",
        "NA",
        "50"
    ]

    valid_ages = []

    for item in raw_data:
        try:
            valid_ages.append(int(item))

        except ValueError:
            print(f"Skipping invalid value: {item}")

    if valid_ages:
        average_age = sum(valid_ages) / len(valid_ages)

        print("Valid Ages:", valid_ages)
        print(f"Average Age: {average_age:.2f}")

    else:
        print("No valid age values found.")


def combined_exceptions():
    """Handle multiple exception types in one except block."""
    print("\n12. MULTIPLE EXCEPTIONS IN ONE BLOCK")

    try:
        number = int(input("Enter number: "))
        result = 100 / number

        print("Result:", result)

    except (ValueError, ZeroDivisionError) as error:
        print("Error:", error)


def main():
    """Run all Day 9 exception-handling demonstrations."""
    print("=" * 60)
    print("DAY 9 - EXCEPTION HANDLING IN PYTHON")
    print("100 DAYS OF DATA SCIENCE")
    print("=" * 60)

    basic_try_except()
    multiple_exceptions()
    exception_object()
    else_block()
    finally_block()
    raise_exception()
    custom_exception()
    nested_exception()
    function_example()
    data_science_example()
    csv_data_cleaning()
    combined_exceptions()

    print("\n" + "=" * 60)
    print("Congratulations!")
    print("You completed Day 9 - Exception Handling.")
    print("=" * 60)


if __name__ == "__main__":
    main()