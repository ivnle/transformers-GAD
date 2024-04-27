import sys, os

def main():
    base_file = sys.argv[1]
    files = sys.argv[2:]

    system_prompt = ("You are an expert in program synthesis. You are tasked with "
                     "solving a Syntax-Guided Synthesis (SyGuS) problem. Your goal is "
                     "to output a function that should produce outputs that satisfy a "
                     "series of constraints when given specific inputs.\n\n")

    content = [system_prompt]

    # Add entries for each selected file
    for f in files:
        # Remove .sl extension and form the expected solution filename
        solution_filename = f[:-3]  # Removes the last three characters '.sl'

        # Check if the solution file exists
        if not os.path.exists(solution_filename):
            print(f"Warning: No solution file found for {solution_filename}. Skipping.")
            continue

        with open(f, 'r') as file:
            question_content = file.read().strip()

        with open(solution_filename, 'r') as file:
            solution_content = file.read().strip()

        content.append(f"Question:\n{question_content}\nSolution:\n{solution_content}\n\n")

    current_file_path = f"correct/{base_file}.sl"
    if os.path.exists(current_file_path):
        with open(current_file_path, 'r') as file:
            question_content = file.read().strip()
        content.append(f"Question:\n{question_content}\nSolution:")
    else:
        print(f"Warning: File {current_file_path} not found. Skipping this as well.")

    print("".join(content))


if __name__ == "__main__":
    main()
