#!/usr/bin/env python3
"""
100-Day Data Science Challenge
Day 1: Data Science Overview & Python Setup

Exercise:
Write a Python script that takes user input for their name and age,
and prints a greeting stating the year they will turn 100.
"""

import datetime
import sys

def get_clean_name() -> str:
    """Prompts the user for their name and ensures it is not empty."""
    while True:
        try:
            name_input = input("Enter your name: ").strip()
            if not name_input:
                print("Error: Name cannot be empty. Please try again.")
                continue
            return name_input
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled by user.")
            sys.exit(0)

def get_valid_age() -> int:
    """Prompts the user for their age, ensuring it is a valid integer between 0 and 120."""
    while True:
        try:
            age_str = input("Enter your age: ").strip()
            if not age_str:
                print("Error: Age cannot be empty. Please enter a number.")
                continue
            
            age = int(age_str)
            if age < 0:
                print("Error: Age cannot be negative. Please enter a valid age.")
            elif age > 120:
                print("Error: That age seems unrealistic (> 120). Please enter a valid age.")
            else:
                return age
        except ValueError:
            print("Error: Invalid input. Please enter a whole number for age.")
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled by user.")
            sys.exit(0)

def calculate_year_100(age: int) -> int:
    """Calculates the calendar year the user will turn 100."""
    current_year = datetime.datetime.now().year
    years_to_100 = 100 - age
    return current_year + years_to_100

def main():
    print("==================================================")
    print("🚀 Day 1 Exercise: Greeting & Age Calculator 🚀")
    print("==================================================")
    
    # Get validated inputs
    name = get_clean_name()
    age = get_valid_age()
    
    # Calculate year
    target_year = calculate_year_100(age)
    
    # Display greeting
    print("\n--------------------------------------------------")
    if age == 100:
        print(f"🎉 Congratulations, {name}! You are already 100 years old this year!")
    elif age > 100:
        past_year = datetime.datetime.now().year - (age - 100)
        print(f"🌟 Hello, {name}! You turned 100 back in the year {past_year}.")
    else:
        print(f"👋 Hello, {name}!")
        print(f"Based on your age of {age}, you will turn 100 years old in the year: {target_year} 🎂")
    print("--------------------------------------------------")
    print("Day 1 Exercise completed successfully!\n")

if __name__ == "__main__":
    main()
