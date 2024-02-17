def reweight_values(output, factor):
    reweighted_output = {}
    for key, value in output.items():
        new_value = round(value * factor)
        reweighted_output[key] = new_value
    return reweighted_output

def check_sum(output):
    return sum(output.values())

if __name__ == "__main__":
    output = {'00101': 24, '01000': 0, '01101': 0, '00000': 0, '00111': 140, '01110': 0, '01010': 0, '00010': 0,
              '00001': 13, '01100': 0, '01001': 0, '00100': 0, '11111': 0, '01011': 17, '00011': 67, '00110': 14,
              '01111': 0} # with 0 all 1
    factor = 1.815 # with 0 all 1

    # output = {'10100': 7, '11010': 0, '11101': 0, '11000': 0, '10110': 109, '00000': 0, '11111': 0, '11110': 0,
    #           '10000': 0, '10101': 28, '11011': 0, '10001': 43, '10011': 54, '10111': 64, '10010': 11, '11001': 0,
    #           '11100': 0} # with 1 all 0
    # factor = 1.5835  # with 1 all 0
    print(check_sum(output))
    reweighted_output = reweight_values(output, factor)
    print(reweighted_output)

    summation = sum(reweighted_output.values())
    print(summation)