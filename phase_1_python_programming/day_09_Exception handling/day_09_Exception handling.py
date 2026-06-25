"""
====================================================
DAY 9 - EXCEPTION HANDLING IN PYTHON
100 DAYS OF DATA SCIENCE
====================================================

Exception Handling allows us to handle runtime errors
without crashing the entire program.

Syntax:

try:
    # risky code
except:
    # handle error
else:
    # runs if no exception occurs
finally:
    # always runs
"""

print("=" * 50)
print("DAY 9 - EXCEPTION HANDLING")
print("=" * 50)

# --------------------------------------------------
# 1. Basic Try Except
# --------------------------------------------------

print("\n1. BASIC TRY EXCEPT")

try:
    number = int(input("Enter a number: "))
    print(f"You entered: {number}")
except ValueError:
    print("Invalid input! Please enter a valid integer.")

# --------------------------------------------------
# 2. Handling Multiple Exceptions
# --------------------------------------------------

print("\n2. MULTIPLE EXCEPTIONS")

try:
    num = int(input("Enter numerator: "))
    den = int(input("Enter denominator: "))

    result = num / den
    print("Result:", result)

except ValueError:
    print("Please enter valid numbers.")

except ZeroDivisionError:
    print("Cannot divide by zero.")

# --------------------------------------------------
# 3. Catching Exception Object
# --------------------------------------------------

print("\n3. EXCEPTION OBJECT")

try:
    x = int(input("Enter a number: "))
    result = 100 / x
    print(result)

except Exception as e:
    print("Error occurred:", e)

# --------------------------------------------------
# 4. Else Block
# --------------------------------------------------

print("\n4. ELSE BLOCK")

try:
    age = int(input("Enter your age: "))
except ValueError:
    print("Age must be a number.")
else:
    print("Input successful.")
    print("Your age is:", age)

# --------------------------------------------------
# 5. Finally Block
# --------------------------------------------------

print("\n5. FINALLY BLOCK")

try:
    file = open("sample.txt", "r")
    content = file.read()
    print(content)

except FileNotFoundError:
    print("File not found.")

finally:
    print("Execution completed.")

# --------------------------------------------------
# 6. Raising Custom Exception
# --------------------------------------------------

print("\n6. RAISE EXCEPTION")

try:
    salary = float(input("Enter salary: "))

    if salary < 0:
        raise ValueError("Salary cannot be negative.")

    print("Salary:", salary)

except ValueError as e:
    print(e)

# --------------------------------------------------
# 7. Custom Exception Class
# --------------------------------------------------

print("\n7. CUSTOM EXCEPTION")


class InvalidAgeError(Exception):
    pass


try:
    age = int(input("Enter age: "))

    if age < 18:
        raise InvalidAgeError(
            "Age must be 18 or above."
        )

    print("Eligible.")

except InvalidAgeError as e:
    print(e)

# --------------------------------------------------
# 8. Nested Exception Handling
# --------------------------------------------------

print("\n8. NESTED TRY EXCEPT")

try:
    try:
        num = int(input("Enter a number: "))
        result = 50 / num
        print(result)

    except ZeroDivisionError:
        print("Inner: Division by zero.")

except ValueError:
    print("Outer: Invalid input.")

# --------------------------------------------------
# 9. User Defined Function Example
# --------------------------------------------------

print("\n9. FUNCTION EXAMPLE")


def divide_numbers(a, b):
    try:
        return a / b

    except ZeroDivisionError:
        return "Cannot divide by zero."


print(divide_numbers(10, 2))
print(divide_numbers(10, 0))

# --------------------------------------------------
# 10. Data Science Example
# --------------------------------------------------

print("\n10. DATA SCIENCE EXAMPLE")

dataset = ["100", "200", "abc", "400", "xyz"]

clean_data = []

for value in dataset:
    try:
        clean_data.append(int(value))

    except ValueError:
        print(f"Invalid data skipped: {value}")

print("Clean Data:", clean_data)

# --------------------------------------------------
# 11. Real World Data Science Scenario
# --------------------------------------------------

print("\n11. CSV DATA CLEANING")

raw_data = [
    "25",
    "30",
    "unknown",
    "45",
    "NA",
    "50"
]

ages = []

for item in raw_data:

    try:
        ages.append(int(item))

    except ValueError:
        print(f"Skipping invalid value: {item}")

average_age = sum(ages) / len(ages)

print("Valid Ages:", ages)
print("Average Age:", average_age)

# --------------------------------------------------
# 12. Multiple Exceptions Together
# --------------------------------------------------

print("\n12. MULTIPLE EXCEPTIONS IN ONE BLOCK")

try:
    num = int(input("Enter number: "))
    result = 100 / num

except (ValueError, ZeroDivisionError) as e:
    print("Error:", e)

# --------------------------------------------------
# END OF DAY 9
# --------------------------------------------------

print("\nCongratulations!")
print("You completed Day 9 - Exception Handling.")