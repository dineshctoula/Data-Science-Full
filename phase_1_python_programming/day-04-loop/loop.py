"""
============================================================
DAY 10 - LOOPS IN PYTHON
100 DAYS OF DATA SCIENCE
============================================================

Topics Covered:
    1. for loop
    2. range()
    3. while loop
    4. break statement
    5. continue statement
    6. nested loops
    7. Number-based loop problems
    8. Sum and average
    9. Multiplication table
    10. Guessing game
    11. Finding the largest number in a dataset
"""


def for_loop_examples():
    """Demonstrate different uses of the for loop."""

    print("\n1. FOR LOOP - range(5)")

    for i in range(5):
        print(i)

    print("\n2. FOR LOOP - range(1, 6)")

    for i in range(1, 6):
        print(i)

    print("\n3. FOR LOOP - range(0, 10, 2)")

    # Starts from 0 and increases by 2.
    for i in range(0, 10, 2):
        print(i)


def while_loop_example():
    """Demonstrate a basic while loop."""

    print("\n4. WHILE LOOP")

    count = 1

    while count <= 5:
        print(count)
        count += 1


def break_example():
    """Demonstrate the break statement."""

    print("\n5. BREAK STATEMENT")

    for i in range(10):

        if i == 5:
            break

        print(i)

    print("Loop stopped when i reached 5.")


def continue_example():
    """Demonstrate the continue statement."""

    print("\n6. CONTINUE STATEMENT")

    for i in range(5):

        if i == 2:
            continue

        print(i)

    print("Number 2 was skipped.")


def nested_loop_example():
    """Demonstrate nested loops."""

    print("\n7. NESTED LOOP")

    for i in range(3):

        # Inner loop runs completely for every
        # iteration of the outer loop.
        for j in range(3):
            print(i, j)


def number_loop_examples():
    """Demonstrate simple number-based loop problems."""

    print("\n8. PRINT NUMBERS FROM 1 TO 20")

    for i in range(1, 21):
        print(i)


def even_numbers():
    """Print even numbers between 1 and 50."""

    print("\n9. EVEN NUMBERS FROM 1 TO 50")

    for i in range(2, 51, 2):
        print(i)


def sum_numbers():
    """Calculate the sum of numbers from 1 to 100."""

    print("\n10. SUM OF NUMBERS FROM 1 TO 100")

    total = 0

    for i in range(1, 101):
        total += i

    print("Sum:", total)


def multiplication_table():
    """Print the multiplication table of a number."""

    print("\n11. MULTIPLICATION TABLE")

    number = 5

    for i in range(1, 11):
        print(f"{number} x {i} = {number * i}")


def guessing_game():
    """Simple number guessing game."""

    print("\n12. GUESSING GAME")

    secret_number = 7

    while True:

        try:
            guess = int(input("Enter your number: "))

            if guess == secret_number:
                print("Correct! You guessed the number.")
                break

            print("Wrong guess. Try again.")

        except ValueError:
            print("Please enter a valid integer.")


def calculate_average():
    """Calculate the average of numbers in a dataset."""

    print("\n13. CALCULATE AVERAGE")

    data = [12, 15, 20, 18, 22]

    total = 0

    for value in data:
        total += value

    average = total / len(data)

    print("Data:", data)
    print("Total:", total)
    print("Average:", average)


def find_largest_number():
    """Find the largest number in a dataset using a loop."""

    print("\n14. FIND LARGEST NUMBER")

    data = [12, 15, 20, 18, 22]

    largest = data[0]

    for value in data:

        if value > largest:
            largest = value

    print("Data:", data)
    print("Largest number:", largest)


def main():
    """Run all Day 10 examples."""

    print("=" * 60)
    print("DAY 10 - LOOPS IN PYTHON")
    print("100 DAYS OF DATA SCIENCE")
    print("=" * 60)

    for_loop_examples()
    while_loop_example()
    break_example()
    continue_example()
    nested_loop_example()
    number_loop_examples()
    even_numbers()
    sum_numbers()
    multiplication_table()
    guessing_game()
    calculate_average()
    find_largest_number()

    print("\n" + "=" * 60)
    print("Congratulations!")
    print("You completed Day 10 - Loops in Python.")
    print("=" * 60)


if __name__ == "__main__":
    main()