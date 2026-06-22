#!/usr/bin/env python3
"""
100-Day Data Science Challenge
Day 8: Functions, Scope & Lambda Expressions

Topics Covered:
1. Function Definition, Parameters, Arguments, Docstrings, and Return Values
2. Variable-length arguments (*args and **kwargs)
3. Scope and LEGB Rule (Local, Enclosing, Global, Built-in)
4. Anonymous Functions (Lambda Expressions)
5. Practical Exercise: Custom Function Application to Collections
"""

def demonstrate_function_basics():
    print("=== 1. FUNCTION DEFINITION & ARGUMENTS ===")
    
    # Basic function with type hints and docstring
    def calculate_compound_interest(principal: float, rate: float, time: int, compounds_per_year: int = 1) -> float:
        """
        Calculates the compound interest for a given principal, rate, time, and compounding frequency.
        
        Args:
            principal (float): The initial sum of money.
            rate (float): The annual interest rate as a decimal (e.g. 0.05 for 5%).
            time (int): The time the money is invested for in years.
            compounds_per_year (int): The number of times interest is compounded per year. Default is 1.
            
        Returns:
            float: The final amount after interest.
        """
        amount = principal * (1 + rate / compounds_per_year) ** (compounds_per_year * time)
        return round(amount, 2)
        
    # Positional arguments
    amount_pos = calculate_compound_interest(1000, 0.05, 5)
    print(f"Positional (1000, 5%, 5 years): ${amount_pos}")
    
    # Keyword arguments
    amount_kw = calculate_compound_interest(principal=1000, rate=0.05, time=5, compounds_per_year=12)
    print(f"Keyword (1000, 5%, 5 years, compounded monthly): ${amount_kw}")
    
    # Docstring access
    print(f"Function Docstring preview:\n{calculate_compound_interest.__doc__.strip().splitlines()[0]}")
    print()


def demonstrate_var_arguments():
    print("=== 2. VARIABLE-LENGTH ARGUMENTS (*args & **kwargs) ===")
    
    # *args collects extra positional arguments into a tuple
    def calculate_mean(*args: float) -> float:
        print(f"  Received args: {args}, type: {type(args)}")
        if not args:
            return 0.0
        return sum(args) / len(args)
        
    mean_val = calculate_mean(10.5, 20.0, 30.5, 40.0)
    print(f"Mean of values: {mean_val}")
    
    # **kwargs collects extra keyword arguments into a dictionary
    def build_user_profile(first_name: str, last_name: str, **kwargs) -> dict:
        print(f"  Received kwargs: {kwargs}, type: {type(kwargs)}")
        profile = {
            "first_name": first_name,
            "last_name": last_name,
            **kwargs
        }
        return profile
        
    profile = build_user_profile("Dinesh", "Toula", job="Data Scientist", country="Nepal", challenge_day=8)
    print(f"User Profile: {profile}")
    print()


# Global variable for scope demonstration
global_variable = "I am Global"

def demonstrate_scope_legb():
    print("=== 3. SCOPE & THE LEGB RULE ===")
    
    # LEGB: Local -> Enclosing -> Global -> Built-in
    
    # 1. Global vs Local
    local_val = "I am Local to demonstrate_scope_legb"
    print(f"Inside function: global_variable = '{global_variable}'")
    print(f"Inside function: local_val = '{local_val}'")
    
    # Modifying a global variable requires the 'global' keyword
    def modify_global():
        global global_variable
        global_variable = "I am modified Global"
        
    modify_global()
    print(f"After modify_global(): global_variable = '{global_variable}'")
    
    # 2. Enclosing Scope & 'nonlocal'
    def outer_function():
        enclosing_val = "I am Enclosing"
        
        def inner_function():
            nonlocal enclosing_val
            # Access enclosing variable
            print(f"    Inner reading enclosing: '{enclosing_val}'")
            
            # Modify enclosing variable
            enclosing_val = "Enclosing has been modified by Inner!"
            
        inner_function()
        print(f"  After inner_function(), enclosing_val = '{enclosing_val}'")
        
    outer_function()
    print()


def demonstrate_lambda_expressions():
    print("=== 4. LAMBDA EXPRESSIONS (ANONYMOUS FUNCTIONS) ===")
    
    # Syntax: lambda arguments: expression
    # Simple lambda
    multiply = lambda x, y: x * y
    print(f"Lambda multiply(5, 6): {multiply(5, 6)}")
    
    # Sorting with a lambda key (very common in data science)
    data_points = [
        {"x": 1, "y": 10},
        {"x": 5, "y": 2},
        {"x": 3, "y": 8}
    ]
    # Sort by 'y' value
    sorted_by_y = sorted(data_points, key=lambda point: point["y"])
    print(f"Original: {data_points}")
    print(f"Sorted by 'y': {sorted_by_y}")
    
    # Using lambda with map and filter
    numbers = [1, 2, 3, 4, 5, 6]
    even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
    squared_numbers = list(map(lambda x: x ** 2, numbers))
    print(f"Original numbers: {numbers}")
    print(f"Filtered evens:   {even_numbers}")
    print(f"Mapped squares:   {squared_numbers}")
    print()


# --- PRACTICAL EXERCISE ---
def run_practical_exercise():
    print("=== 5. PRACTICAL EXERCISE ===")
    
    # 1. Lambda function to square a number
    square_lambda = lambda x: x ** 2
    print(f"square_lambda(9): {square_lambda(9)}")
    assert square_lambda(9) == 81, "Lambda square function failed!"
    
    # 2. Regular function that applies a mathematical formula function/lambda to a list
    def apply_formula_to_list(numbers: list, formula_func) -> list:
        """
        Applies a custom mathematical formula function/lambda to each element in a list.
        
        Args:
            numbers (list): List of numeric values.
            formula_func (function/lambda): A callable that takes a single number and returns a number.
            
        Returns:
            list: A new list with the formula applied to each element.
        """
        # We can implement this with a loop or list comprehension
        return [formula_func(x) for x in numbers]
        
    test_list = [1, 2, 3, 4, 5]
    
    # Apply square formula using square_lambda
    squares = apply_formula_to_list(test_list, square_lambda)
    print(f"Applying square formula to {test_list} -> {squares}")
    assert squares == [1, 4, 9, 16, 25], "apply_formula_to_list with square failed!"
    
    # Apply another custom formula: f(x) = 2x + 10
    linear_transform = apply_formula_to_list(test_list, lambda x: 2 * x + 10)
    print(f"Applying '2x + 10' formula to {test_list} -> {linear_transform}")
    assert linear_transform == [12, 14, 16, 18, 20], "apply_formula_to_list with custom linear formula failed!"
    
    print("\nExercise completed successfully! All assertions passed.")


if __name__ == "__main__":
    demonstrate_function_basics()
    demonstrate_var_arguments()
    demonstrate_scope_legb()
    demonstrate_lambda_expressions()
    run_practical_exercise()
