#!/usr/bin/env python3
"""
100-Day Data Science Challenge
Day 7: Tuples, Mutability & Memory References

Topics Covered:
1. Tuples Basics (Creation, Singleton, Indexing, Slicing, Immutability)
2. Tuple Packing, Unpacking, and Extended Unpacking (*)
3. Mutability vs Immutability (Core Types)
4. Memory References, Identity (is) vs Equality (==)
5. Small Integer Interning & Referential Immutability
6. Shallow Copy vs Deep Copy (copy module)
7. Pass-by-Object-Reference in Functions
8. Practical Exercise: Solving a Shared State Config Bug
"""

import copy

def demonstrate_tuples_basics():
    print("=== 1. TUPLES BASICS ===")
    # Creation
    tup1 = (1, 2, 3)
    tup2 = "a", "b", "c" # Parentheses are optional but recommended
    print(f"tup1: {tup1}, type: {type(tup1)}")
    print(f"tup2: {tup2}, type: {type(tup2)}")
    
    # CRITICAL: Singleton tuple (single element)
    not_a_tuple = (5) # This is just an integer!
    singleton_tuple = (5,) # Notice the trailing comma
    print(f"not_a_tuple: {not_a_tuple}, type: {type(not_a_tuple)}")
    print(f"singleton_tuple: {singleton_tuple}, type: {type(singleton_tuple)}")
    
    # Indexing and Slicing (works like lists)
    print(f"First element: {tup1[0]}")
    print(f"Last two: {tup1[-2:]}")
    
    # Immutability
    print("\nAttempting to modify a tuple:")
    try:
        tup1[0] = 99
    except TypeError as e:
        print(f"  TypeError caught successfully: {e}")
        
    # Tuple methods: count and index
    tup3 = (1, 2, 3, 2, 4, 2)
    print(f"tup3: {tup3}")
    print(f"Count of 2: {tup3.count(2)}")
    print(f"First index of 2: {tup3.index(2)}")
    print(f"First index of 2 after index 2: {tup3.index(2, 2)}")
    print()


def demonstrate_packing_unpacking():
    print("=== 2. TUPLE PACKING & UNPACKING ===")
    # Packing
    packed = 10, "data", 3.14
    print(f"Packed tuple: {packed}")
    
    # Unpacking
    x, y, z = packed
    print(f"Unpacked: x={x}, y='{y}', z={z}")
    
    # Variable swapping using tuples
    a, b = 5, 10
    print(f"Before swap: a={a}, b={b}")
    a, b = b, a # Packing right-side, unpacking to left-side
    print(f"After swap: a={a}, b={b}")
    
    # Extended unpacking with *
    numbers = (1, 2, 3, 4, 5, 6)
    first, *middle, last = numbers
    print(f"Numbers: {numbers}")
    print(f"first: {first}")
    print(f"middle (always a list): {middle}")
    print(f"last: {last}")
    print()


def demonstrate_mutability_identity():
    print("=== 3. MUTABILITY VS IMMUTABILITY & IDENTITY ===")
    # Immutable: int, float, str, tuple, bool
    # Mutable: list, dict, set
    
    # Equality (==) vs Identity (is)
    # == checks if values are equal
    # is checks if they point to the exact same memory location (same id)
    
    list_a = [1, 2, 3]
    list_b = [1, 2, 3]
    list_c = list_a
    
    print(f"list_a: {list_a} (id: {id(list_a)})")
    print(f"list_b: {list_b} (id: {id(list_b)})")
    print(f"list_c: {list_c} (id: {id(list_c)})")
    
    print(f"list_a == list_b: {list_a == list_b} (Same value)")
    print(f"list_a is list_b: {list_a is list_b} (Same memory reference?)")
    print(f"list_a is list_c: {list_a is list_c} (Same memory reference?)")
    
    # Modifying a shared reference affects all references!
    list_c.append(4)
    print(f"After modifying list_c: list_a is {list_a}")
    print()


def demonstrate_interning_and_referential():
    print("=== 4. SMALL INTEGER INTERNING & REFERENTIAL IMMUTABILITY ===")
    # Python caches small integers (-5 to 256) for performance optimization
    int_a = 250
    int_b = 250
    print(f"int_a (250) is int_b (250): {int_a is int_b} (cached)")
    
    int_c = 300
    int_d = 300
    print(f"int_c (300) is int_d (300): {int_c is int_d} (may vary by python implementation/session, usually False in REPL, True in same compilation unit/script)")
    
    # Referential Immutability
    # A tuple is immutable, meaning its references cannot change.
    # However, if one of those references points to a mutable object,
    # the mutable object CAN still be modified.
    
    nested_list = [1, 2]
    mixed_tuple = (10, nested_list, "hello")
    print(f"Initial mixed_tuple: {mixed_tuple}")
    
    # Trying to change the reference itself raises TypeError
    try:
        mixed_tuple[1] = [3, 4]
    except TypeError as e:
        print(f"  Cannot change reference: {e}")
        
    # BUT we can modify the contents of the mutable object inside
    mixed_tuple[1].append(3)
    print(f"After modifying inner list: {mixed_tuple}")
    print()


def demonstrate_copies():
    print("=== 5. SHALLOW COPY VS DEEP COPY ===")
    original = [[1, 2, 3], [4, 5, 6]]
    
    # Shallow Copy
    shallow_copied = copy.copy(original)
    # Deep Copy
    deep_copied = copy.deepcopy(original)
    
    print(f"Original:       {original} (id: {id(original)}, elements id: {[id(x) for x in original]})")
    print(f"Shallow Copy:   {shallow_copied} (id: {id(shallow_copied)}, elements id: {[id(x) for x in shallow_copied]})")
    print(f"Deep Copy:      {deep_copied} (id: {id(deep_copied)}, elements id: {[id(x) for x in deep_copied]})")
    
    # Modifying the top level of shallow copy doesn't affect original
    shallow_copied.append([7, 8, 9])
    # Modifying a nested mutable element of shallow copy DOES affect original!
    shallow_copied[0].append(99)
    
    # Modifying deep copy has absolutely no effect on original
    deep_copied[1].append(999)
    
    print(f"\nAfter Modifications:")
    print(f"Original:       {original}")
    print(f"Shallow Copy:   {shallow_copied}")
    print(f"Deep Copy:      {deep_copied}")
    print()


def modify_data(num, lst):
    # num is immutable. Rebinding the local variable num doesn't change the caller's variable.
    num = 100
    # lst is mutable. Modifying it in-place affects the caller's list.
    lst.append("new_element")
    # Rebinding lst to a new list doesn't affect caller's reference.
    lst = [9, 9, 9]

def demonstrate_pass_by_reference():
    print("=== 6. PASS-BY-OBJECT-REFERENCE ===")
    my_num = 10
    my_list = [1, 2]
    print(f"Before function: my_num={my_num}, my_list={my_list}")
    modify_data(my_num, my_list)
    print(f"After function:  my_num={my_num}, my_list={my_list}")
    print()


# --- PRACTICAL EXERCISE ---
class ConfigManager:
    """
    Simulates a configuration manager system.
    Demonstrates the bug of sharing mutable default dictionary state,
    and how deep copying solves it.
    """
    def __init__(self, default_config):
        # BUGGY version: self.config = default_config
        # This makes all ConfigManager instances share the same nested dictionary structures if modified.
        # SAFE version uses deepcopy.
        self.config = copy.deepcopy(default_config)
        
    def update_database_url(self, new_url):
        self.config["database"]["url"] = new_url
        
    def get_config(self):
        return self.config

def run_practical_exercise():
    print("=== 7. PRACTICAL EXERCISE: CONFIG SYSTEM SHALLOW REFERENCE BUG ===")
    
    default_settings = {
        "app_name": "DataScienceApp",
        "debug": False,
        "database": {
            "host": "localhost",
            "port": 5432,
            "url": "postgresql://localhost/db"
        }
    }
    
    print("1. Creating config instances using standard assignment (sharing references)...")
    # Let's simulate what happens if we did NOT use deepcopy
    bad_config1 = default_settings
    # If we modify database URL in bad_config1:
    bad_config1["database"]["url"] = "postgresql://prod_server/db"
    
    print(f"  default_settings url: {default_settings['database']['url']} (Changed!)")
    
    # Resetting default_settings
    default_settings["database"]["url"] = "postgresql://localhost/db"
    
    print("\n2. Creating config instances using ConfigManager (with deepcopy)...")
    manager_dev = ConfigManager(default_settings)
    manager_prod = ConfigManager(default_settings)
    
    print("Updating manager_prod url...")
    manager_prod.update_database_url("postgresql://prod_db:5432/production")
    
    print(f"  dev config url:  {manager_dev.get_config()['database']['url']}")
    print(f"  prod config url: {manager_prod.get_config()['database']['url']}")
    print(f"  default settings: {default_settings['database']['url']} (Remained unchanged!)")
    print("\nDeep Copy successfully isolated the state between different configuration environments!")


if __name__ == "__main__":
    demonstrate_tuples_basics()
    demonstrate_packing_unpacking()
    demonstrate_mutability_identity()
    demonstrate_interning_and_referential()
    demonstrate_copies()
    demonstrate_pass_by_reference()
    run_practical_exercise()
