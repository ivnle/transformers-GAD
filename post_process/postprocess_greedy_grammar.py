from tqdm import tqdm
import re
import logging

def postprocess_greedy_w_no_explain_depreciated(input_file):
    # Define the set of strings for G*
    G_star_set = {'11011', '10010', '00000', '11100', '10110', '10000', '10111', '11101', '11111', '10001', '10101',
                  '10100', '10011', '11110', '11010', '11000', '11001'}

    # Initialize counters
    G_star = {key: 0 for key in G_star_set}  # G*
    G_prime = {f"{i:05b}": 0 for i in range(32) if f"{i:05b}" not in G_star_set}  # G'
    G = {f"{i:05b}": 0 for i in range(32)}
    R = {}  # Rejected strings, not fitting G* or G'

    # Compile the regex pattern for extracting binary strings
    pattern = re.compile(r'\n\n([01]{5})|\n([01]{5})\b')

    def process_log_entry(entry):
        # Check if the entry matches the expected pattern
        match = pattern.search(entry)
        print(f"match: {match}")
        if match:
            # Extract the binary string (handling either group matched)
            binary_string = match.group(1) or match.group(2)
            binary_string = str(binary_string)
            print(f"binary_string: {binary_string}")
            return binary_string
        return "rejected"

    # Process the log file
    total_count = 0  # Total count of strings processed
    G_star_count = 0  # Total count of strings in G (G* + G')
    G_prime_count = 0  # Total count of strings in G'
    G_count = 0
    R_count = 0

    with open(input_file, 'r') as file:
        print(f"total_count: {total_count}")
        for line in file:
            binary_string = process_log_entry(line)
            if binary_string != "rejected":
                if binary_string in G_star:
                    G_star[binary_string] += 1
                    G_star_count += 1
                    if G_star_count >= 500:  # Stop if G* reaches 500
                        break
                else:
                    G_prime[binary_string] += 1
                    G_prime_count += 1
                G[binary_string] += 1
                G_count += 1
            else:
                R[binary_string] = R.get(binary_string, 0) + 1
                R_count += 1
            total_count += 1
            if total_count % 10 == 0:
                print(f"G*: {G_star}")
                print(f"G': {G_prime}")
                print(f"R: {R}")
    # Calculate the summary
    G_check_count = sum(G_star.values()) + sum(G_prime.values())  # Total G (G* + G')
    G_star_check_count = sum(G_star.values())  # Total G*
    G_prime_check_count = sum(G_prime.values())  # Total G'
    print(f"G*: {G_star}")
    print(f"G': {G_prime}")
    print(f"R: {R}")
    print(f"total_count: {total_count}")
    print(f"G_count: {G_count}")
    print(f"G_star_count: {G_star_count}")
    print(f"G_prime_count: {G_prime_count}")
    print(f"G_check_count: {G_check_count}")
    print(f"G_star_check_count: {G_star_check_count}")
    print(f"G_prime_check_count: {G_prime_check_count}")
    print(f"R_count: {R_count}")


    return G_star, G_prime, R, total_count, G_count, G_star_count, G_prime_count, G_check_count, G_star_check_count, G_prime_check_count, R_count


def postprocess_greedy_w_no_explain(input_file, log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    # Define the regex pattern to match binary strings after the prompt
    # This pattern looks for the prompt, followed by optional space, newline(s), and captures the 5-character binary string
    pattern = re.compile(
        r"Directly show the generated string without explanation\.\s*\\n(\\n)?([01]{5})")

    # Initialize dictionaries for G*, G', and R (rejected)
    G_star_set = {'11011', '10010', '00000', '11100', '10110', '10000', '10111', '11101', '11111', '10001', '10101',
                  '10100', '10011', '11110', '11010', '11000', '11001'}
    G_star = {key: 0 for key in G_star_set}  # G*
    G_prime = {f"{i:05b}": 0 for i in range(32) if f"{i:05b}" not in G_star_set}  # G'
    R = {'rejected': 0}  # Rejected strings

    total_count = 0  # Total count of strings processed
    G_star_count = 0  # Total count of strings in G (G* + G')
    G_prime_count = 0  # Total count of strings in G'
    G_count = 0

    with open(input_file, 'r') as file:
        for line in file:
            if G_star_count >= 500:
                break
            match = pattern.search(line)
            if match:
                binary_string = match.group(2)  # Extract the binary string
                print(f"binary_string: {binary_string}")
                if binary_string in G_star:
                    G_star[binary_string] += 1
                    G_star_count += 1
                elif binary_string in G_prime:
                    G_prime[binary_string] += 1
                    G_prime_count += 1
                G_count += 1
            else:
                # Increment the rejected count if the line doesn't match the expected format
                R['rejected'] += 1
            total_count += 1

            if G_count % 10 == 0:
                logging.info(f"G_count: {G_count}")
                logging.info(f"G*: {G_star}")
                logging.info(f"G': {G_prime}")
                print(f"R: {R}")

    # Print summaries
    logging.info(f"G* (Specified Set): {G_star}")
    logging.info(f"G' (Other Valid Binary Strings): {G_prime}")
    logging.info(f"R (Rejected): {R}")
    logging.info(f"Total Count: {total_count}")
    logging.info(f"G Count: {G_count}")
    logging.info(f"G* Count: {G_star_count}")
    logging.info(f"G' Count: {G_prime_count}")

    return G_star, G_prime, R, total_count

if __name__ == "__main__":
    # pattern = re.compile(
    #     r"Generate a random binary string of length 5\? Directly show the generated string without explanation\.\s*\\n(\\n)?([01]{5})")
    # pattern = re.compile(
    #     r"Be a helpful assistant\. Generate a random binary string of length 5 following the grammar: root ::= '00000' | '1's; s ::= '0' | '1' | '0's | '1's\? Directly show the generated string without explanation\.\s*\\n(\\n)?([01]{5})"
    # G_star, G_prime, R, total_count = postprocess_greedy_w_no_explain('postprocess_prompt_no_explain.txt', 'postprocess_log_no_explain.txt')

    G_star, G_prime, R, total_count = postprocess_greedy_w_no_explain('data2process/postprocess_prompt_grammar.txt',
                                                                      'log/postprocess_log_grammar.txt')
    # def process_log_entry(entry):
    #     # Check if the entry matches the expected pattern
    #     match = pattern.search(entry)
    #     print(f"match: {match}")
    #     if match:
    #         # Extract the binary string (handling either group matched)
    #         binary_string = match.group(1) or match.group(2)
    #         print(f"binary_string: {binary_string}")
    #         return binary_string
    #     return None

    # pattern = re.compile(r'\n\n([01]{5})|\n([01]{5})\b')
    # line = "2024-02-05 01:52:16,489:INFO:greedy generations: ['Be a helpful assistant. Generate a random binary string of length 5? Directly show the generated string without explanation.\n\n11011']"
    # binary_string = process_log_entry(line)