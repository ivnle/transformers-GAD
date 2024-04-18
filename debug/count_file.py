import os

def count_sl_files(directory):
    # Initialize a counter for files ending with .sl
    count = 0
    # Loop through each file in the specified directory
    for file in os.listdir(directory):
        # Check if the file ends with .sl
        if file.endswith('.sl'):
            count += 1
    return count

# Directory path
directory_path = 'correct'

# Calling the function and printing the result
num_sl_files = count_sl_files(directory_path)
print(f"There are {num_sl_files} files ending with .sl in the directory '{directory_path}'.")
