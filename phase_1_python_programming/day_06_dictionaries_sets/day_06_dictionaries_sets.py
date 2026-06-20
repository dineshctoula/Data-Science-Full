#!/usr/bin/env python3
"""
100-Day Data Science Challenge
Day 6: Dictionaries & Sets

Topics Covered:
1. Dictionary Basics (Creation, Accessing, Modifying, Adding items)
2. Dictionary Methods (.get(), .keys(), .values(), .items(), .pop(), .update())
3. Sets (Uniqueness, Creation, adding/removing items)
4. Set Operations (Union, Intersection, Difference, Symmetric Difference)
5. Practical Exercise: Word Frequency Counter
"""

import string

def demonstrate_dictionaries():
    print("--- 1. DICTIONARY BASICS ---")
    # Creation
    student = {
        "name": "Dinesh",
        "age": 22,
        "courses": ["Python", "Data Science"],
        "is_active": True
    }
    print(f"Initial dictionary: {student}")
    
    # Accessing values
    # Direct access (raises KeyError if key doesn't exist)
    print(f"Name: {student['name']}")
    
    # Using .get() (returns None or default value if key doesn't exist)
    email = student.get("email")
    print(f"Email (non-existent key): {email}")
    email_with_default = student.get("email", "not_provided@example.com")
    print(f"Email with default: {email_with_default}")
    
    # Modifying & Adding items
    student["age"] = 23
    student["email"] = "dinesh@example.com"
    print(f"Modified dictionary: {student}")
    
    print("\n--- 2. DICTIONARY METHODS ---")
    # keys(), values(), items()
    print(f"Keys: {list(student.keys())}")
    print(f"Values: {list(student.values())}")
    print(f"Items (key-value pairs): {list(student.items())}")
    
    # Iteration
    print("Iterating through key-value pairs:")
    for key, val in student.items():
        print(f"  {key}: {val}")
        
    # pop() and popitem()
    removed_val = student.pop("is_active")
    print(f"Removed 'is_active' value: {removed_val}")
    print(f"Dictionary after pop: {student}")
    
    last_item = student.popitem()
    print(f"Removed last item: {last_item}")
    print(f"Dictionary after popitem: {student}")
    
    # update() method
    student.update({"age": 24, "location": "Kathmandu", "gpa": 3.9})
    print(f"Dictionary after update(): {student}")


def demonstrate_sets():
    print("\n--- 3. SETS BASICS ---")
    # Creation and Uniqueness
    numbers = {1, 2, 3, 4, 4, 5, 2, 1}
    print(f"Set of numbers (duplicates removed automatically): {numbers}")
    
    # Adding and removing
    numbers.add(6)
    print(f"Set after adding 6: {numbers}")
    
    numbers.remove(3) # Raises KeyError if 3 isn't present
    print(f"Set after removing 3: {numbers}")
    
    numbers.discard(10) # Safe removal, does not raise error if 10 isn't present
    print(f"Set after discarding 10: {numbers}")
    
    print("\n--- 4. SET OPERATIONS ---")
    set_a = {1, 2, 3, 4, 5}
    set_b = {4, 5, 6, 7, 8}
    print(f"Set A: {set_a}")
    print(f"Set B: {set_b}")
    
    # Union (|)
    union_set = set_a | set_b
    print(f"Union (A | B): {union_set}")
    
    # Intersection (&)
    intersection_set = set_a & set_b
    print(f"Intersection (A & B): {intersection_set}")
    
    # Difference (-)
    difference_a_b = set_a - set_b
    print(f"Difference (A - B): {difference_a_b}")
    
    # Symmetric Difference (^)
    symmetric_diff = set_a ^ set_b
    print(f"Symmetric Difference (A ^ B): {symmetric_diff}")


def word_frequency_counter(paragraph):
    print("\n--- 5. EXERCISE: WORD FREQUENCY COUNTER ---")
    # Clean the text: remove punctuation and convert to lowercase
    # Using string.punctuation: '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    cleaned_paragraph = ""
    for char in paragraph:
        if char not in string.punctuation:
            cleaned_paragraph += char
        else:
            cleaned_paragraph += " " # Replace punctuation with a space to prevent merging words
            
    words = cleaned_paragraph.lower().split()
    
    # Count occurrences
    freq_dict = {}
    for word in words:
        freq_dict[word] = freq_dict.get(word, 0) + 1
        
    return freq_dict


if __name__ == "__main__":
    demonstrate_dictionaries()
    demonstrate_sets()
    
    # Exercise paragraph
    sample_text = (
        "Data science is an interdisciplinary academic field that uses statistics, "
        "scientific computing, scientific methods, processes, algorithms and systems "
        "to extract or extrapolate knowledge and insights from noisy, structured, "
        "and unstructured data. Data science is multifaceted!"
    )
    
    word_counts = word_frequency_counter(sample_text)
    
    # Sort and display the top 5 most common words
    sorted_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    
    print(f"Sample Text:\n\"{sample_text}\"")
    print("\nWord frequencies (Alphabetical):")
    for word in sorted(word_counts.keys()):
        print(f"  '{word}': {word_counts[word]}")
        
    print("\nTop 5 most common words:")
    for word, count in sorted_counts[:5]:
        print(f"  '{word}': {count}")
