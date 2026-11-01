#!/usr/bin/env python3
"""
100-Day Data Science Challenge
Day 5: Lists & List Comprehensions

Topics Covered:
1. List Creation, Indexing, and Slicing (positive, negative, steps)
2. List Methods (append, extend, insert, remove, pop, sort, reverse, index, count)
3. Mutability and Copying (references vs. shallow copy vs. deep copy)
4. List Comprehensions (basic, with filtering 'if', with mapping 'if-else')
5. Practical Exercise: Squares of even numbers
"""

import copy

def demonstrate_list_basics():
    print("\n--- 1. List Creation, Indexing & Slicing ---")
    
    # Creation
    fruits = ["apple", "banana", "cherry", "date", "elderberry"]
    mixed = [42, "hello", 3.14, True, [1, 2]]
    print(f"Fruits list: {fruits}")
    print(f"Mixed list: {mixed}")
    
    # Indexing (Positive and Negative)
    print(f"First element (fruits[0]): {fruits[0]}")
    print(f"Last element (fruits[-1]): {fruits[-1]}")
    print(f"Second to last (fruits[-2]): {fruits[-2]}")
    
    # Slicing [start:stop:step]
    print(f"Slice fruits[1:4]: {fruits[1:4]} (elements at index 1, 2, 3)")
    print(f"Slice fruits[:3]: {fruits[:3]} (first 3 elements)")
    print(f"Slice fruits[2:]: {fruits[2:]} (from index 2 to end)")
    print(f"Every second fruit (fruits[::2]): {fruits[::2]}")
    print(f"Reversed fruits (fruits[::-1]): {fruits[::-1]}")


def demonstrate_list_methods():
    print("\n--- 2. List Methods & Operations ---")
    numbers = [10, 20, 30, 20, 40]
    print(f"Initial list: {numbers}")
    
    # append(item)
    numbers.append(50)
    print(f"After append(50): {numbers}")
    
    # extend(iterable)
    numbers.extend([60, 70])
    print(f"After extend([60, 70]): {numbers}")
    
    # insert(index, item)
    numbers.insert(2, 25)
    print(f"After insert(2, 25): {numbers}")
    
    # remove(item) - removes first occurrence
    numbers.remove(20)
    print(f"After remove(20): {numbers}")
    
    # pop(index) - removes and returns item (default last)
    popped = numbers.pop()
    print(f"After pop() (popped {popped}): {numbers}")
    popped_idx = numbers.pop(1)
    print(f"After pop(1) (popped {popped_idx}): {numbers}")
    
    # index(item) and count(item)
    print(f"Index of 30: {numbers.index(30)}")
    print(f"Count of 20: {numbers.count(20)}")
    
    # sort() and reverse()
    numbers.sort()
    print(f"After sort(): {numbers}")
    numbers.reverse()
    print(f"After reverse(): {numbers}")


def demonstrate_mutability_copying():
    print("\n--- 3. Mutability & Copying ---")
    
    # Reference assignment (modifying one affects the other)
    original = [1, 2, [3, 4]]
    ref_copy = original
    print(f"Original: {original}")
    print(f"Ref Copy: {ref_copy}")
    
    ref_copy[0] = 99
    print("Modifying ref_copy[0] = 99...")
    print(f"Original: {original} (Changed!)")
    print(f"Ref Copy: {ref_copy}")
    
    # Shallow Copy (new outer list, but nested items are still references)
    original[0] = 1 # reset
    shallow = original.copy() # or list(original) or original[:]
    print("\nCreating shallow copy...")
    shallow[0] = 88
    print("Modifying shallow[0] = 88 (outer list value)...")
    print(f"Original: {original} (Unchanged)")
    print(f"Shallow: {shallow}")
    
    shallow[2][0] = 77
    print("Modifying shallow[2][0] = 77 (nested list value)...")
    print(f"Original: {original} (Changed!)")
    print(f"Shallow: {shallow}")
    
    # Deep Copy (completely independent)
    original[2][0] = 3 # reset
    deep = copy.deepcopy(original)
    print("\nCreating deep copy...")
    deep[2][0] = 55
    print("Modifying deep[2][0] = 55 (nested list value)...")
    print(f"Original: {original} (Unchanged!)")
    print(f"Deep Copy: {deep}")


def demonstrate_list_comprehensions():
    print("\n--- 4. List Comprehensions ---")
    
    # Standard loop vs List Comprehension
    # Goal: Squares of numbers 0 to 5
    squares_loop = []
    for x in range(6):
        squares_loop.append(x**2)
    print(f"Squares using for-loop:         {squares_loop}")
    
    squares_comp = [x**2 for x in range(6)]
    print(f"Squares using comprehension:    {squares_comp}")
    
    # Comprehension with condition (filtering)
    # syntax: [expression for item in iterable if condition]
    even_squares = [x**2 for x in range(10) if x % 2 == 0]
    print(f"Even squares (0-9):            {even_squares}")
    
    # Comprehension with if-else (mapping)
    # syntax: [expression_if_true if condition else expression_if_false for item in iterable]
    labels = ["Even" if x % 2 == 0 else "Odd" for x in range(6)]
    print(f"Labels (0-5):                   {labels}")


def run_exercise():
    print("\n=== Day 5 Exercise: Squares of Even Numbers ===")
    
    # Define a sample list of numbers
    input_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    print(f"Input List: {input_numbers}")
    
    # Use List Comprehension to filter even numbers and square them
    even_squares = [num**2 for num in input_numbers if num % 2 == 0]
    
    print(f"Result (Squares of even numbers): {even_squares}")
    print("=================================================")


def main():
    print("==================================================")
    print("🚀 Day 5: Lists & List Comprehensions 🚀")
    print("==================================================")
    
    demonstrate_list_basics()
    demonstrate_list_methods()
    demonstrate_mutability_copying()
    demonstrate_list_comprehensions()
    run_exercise()

if __name__ == "__main__":
    main()
