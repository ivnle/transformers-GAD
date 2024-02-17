import re
import logging

def select_G_star(output):
    # Define the keys for which we want to select values from output
    # G_star_keys = ['10100', '11010', '11101', '11000', '10110', '00000', '11111', '11110',
    #                '10000', '10101', '11011', '10001', '10011', '10111', '10010', '11001', '11100'] # star with 1 all 0

    G_star_keys = ['00101', '01000', '01101', '00000', '00111', '01110', '01010', '00010', '00001', '01100', '01001', '00100',
     '11111', '01011', '00011', '00110', '01111'] # G_star_star_keys, start with 0 all 1

    # Create a new dictionary for G_star that only includes keys from output that exist in G_star_keys
    G_star = {key: output[key] for key in G_star_keys if key in output}

    return G_star

def postprocess_01(output, input_file, log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    G_star = select_G_star(output)
    G_star_count = sum(G_star.values())
    G_count = sum(output.values())

    pattern = re.compile(r'(?<=\?)[01]+')

    with open(input_file, 'r') as file:
        for line in file:
            if G_star_count >= 500:
                break
            # Find all binary strings in the line
            matches = pattern.findall(line)
            for binary_str in matches:
                print(binary_str)
                # Update output dictionary
                if binary_str in output:
                    output[binary_str] += 1
                else:
                    output['other'] += 1
                G_count += 1
                if binary_str in G_star:
                    G_star[binary_str] += 1
                    G_star_count += 1
    G_prime_count = G_count - G_star_count
    logging.info(f"G* (Specified Set): {G_star}")
    logging.info(f"G: {output}")
    logging.info(f"G Count: {G_count}")
    logging.info(f"G* Count: {G_star_count}")
    logging.info(f"G' Count: {G_prime_count}")

    return G_star, G_star_count, G_count, G_prime_count, output


if __name__ == "__main__":
    # output = {'00000': 0, '00001': 0, '00010': 0, '00011': 0, '00100': 0, '00101': 0, '00110': 0, '00111': 0,
    #                  '01000': 0, '01001': 6, '01010': 4, '01011': 37, '01100': 13, '01101': 32, '01110': 28,
    #                  '01111': 10, '10000': 15, '10001': 28, '10010': 35, '10011': 39, '10100': 18, '10101': 51,
    #                  '10110': 83, '10111': 46, '11000': 3, '11001': 8, '11010': 18, '11011': 16, '11100': 3, '11101': 6,
    #                  '11110': 1, '11111': 0, 'other': 0} # first 500 outputs from GCD_01

    # output = {'00000': 0, '00001': 0, '00010': 0, '00011': 0, '00100': 0, '00101': 0, '00110': 0, '00111': 0, '01000': 0,
    #     '01001': 14, '01010': 13, '01011': 46, '01100': 15, '01101': 43, '01110': 35, '01111': 15, '10000': 22,
    #     '10001': 37, '10010': 52, '10011': 46, '10100': 28, '10101': 68, '10110': 114, '10111': 61, '11000': 4,
    #     '11001': 11, '11010': 24, '11011': 22, '11100': 4, '11101': 6, '11110': 1, '11111': 0, 'other': 0}

    # output = {'00000': 0, '00001': 6, '00010': 0, '00011': 19, '00100': 0, '00101': 10, '00110': 4, '00111': 48, '01000': 0,
    #  '01001': 0, '01010': 0, '01011': 7, '01100': 0, '01101': 0, '01110': 0, '01111': 0, '10000': 0, '10001': 13,
    #  '10010': 9, '10011': 23, '10100': 3, '10101': 9, '10110': 33, '10111': 19, '11000': 0, '11001': 0, '11010': 0,
    #  '11011': 0, '11100': 0, '11101': 0, '11110': 0, '11111': 0, 'other': 797} # first 1000 outputs from GCD_01 bit

    output = {'01000': 0, '00110': 3, '00111': 16, '00010': 0, '01101': 0, '01100': 0, '11111': 127, '00011': 12,
              '00000': 0, '01010': 0, '01011': 4, '00100': 0, '00101': 3, '01110': 0, '00001': 3, '01111': 0,
              '01001': 0, 'other': 332} # with 0 all 1

    # output = {'11011': 0, '10010': 1, '11010': 0, '10110': 16, '11101': 0, '10001': 3, '10101': 6, '10000': 0,
    #           '00000': 384, '10100': 1, '10011': 3, '11100': 0, '11111': 0, '11110': 0, '11001': 0, '10111': 9,
    #           '11000': 0, 'other': 77} # with 1 all 0

    # input_file = 'data2process/postprocess_GCD_01_bit.txt'
    input_file = 'data2process/postprocess_GCD_start_w_0_all_1_bit.txt'
    # input_file = 'data2process/postprocess_GCD_start_w_1_all_0_bit.txt'
    log_file = 'log/postprocess_log_GCD_bit.txt'
    G_star, G_star_count, G_count, G_prime_count, output = postprocess_01(output, input_file, log_file)
    count = sum(output.values())
    G_star = select_G_star(output)
    G_star_count = sum(G_star.values())
    print(count, G_star_count)

    # my_dict = {'00101': 0, '01000': 0, '01101': 29, '00000': 0, '00111': 0, '01110': 29, '01010': 20,
    #            '00010': 0, '00001': 0, '01100': 11, '01001': 5, '00100': 0, '11111': 365, '01011': 29,
    #            '00011': 0, '00110': 0, '01111': 12}
    # keys_list = list(my_dict.keys())
    # print(keys_list)
