"""
Check if the given string is a valid string according to the given grammar,
used for rejection sampling.
TODO: integrate to strictly grammar following version
"""

def is_valid_string_start_w_1_all_0(s: str) -> bool:
    if s == "00000":
        return True
    elif s.startswith('1') and all(char in '01' for char in s):
        return True
    else:
        return False

def is_valid_string_0(s: str) -> bool:
    # Check if empty or contains any character other than '0' or '1'
    if not s or any(c not in '01' for c in s):
        return False

    # Check if the s follows the pattern of one or more '1's optionally followed by a single '0'
    if s == '0':  # Single '0' is valid
        return True
    if s.endswith('0'):  # If it ends with '0', the rest must be '1's
        return all(c == '1' for c in s[:-1])
    else:  # If it doesn't end with '0', all characters must be '1's
        return all(c == '1' for c in s)

def is_valid_string_1(s: str) -> bool:
    # Check if the s is empty or contains any character other than '0' or '1'
    if not s or any(c not in '01' for c in s):
        return False

    # A valid s can only end with a single '1' and have '0's before it (if any)
    if s.endswith('1'):
        return all(c == '0' for c in s[:-1])
    else:
        return all(c == '0' for c in s)

def is_valid_string_01(s: str) -> bool:
    if not s or any(c not in '01' for c in s):
        return False
    return True