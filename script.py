import subprocess
import matplotlib.pyplot as plt
import numpy as np
import get_desired_string_dict
from get_desired_string_dict import stringsofLenk
import json

# model = './models/7B/ggml-model-q4_0.gguf'
grammar = './examples/grammars/string_01.ebnf'
iterations = 100
k = 2
prompt = 'Generate a random binary string of length at most {k}?'.format(k=k)
Temperature = "0.8"

# args = ['./main', '-m', model, '-n', str(k), '-p', prompt, '--grammar-file', grammar, '--color', '--temp', Temperature]

f = open(grammar)
s = f.read()
f.close()
output = stringsofLenk(s, k)
ideal = stringsofLenk(s, k)
for i in ideal.keys():
    ideal[i] = round(iterations / len(ideal.keys()))
faithful = stringsofLenk(s, k)
output['other'] = 0
ideal['other'] = 0

for i in range(iterations):
    result = subprocess.run(args, capture_output=True, text=True, shell=False)

    res = result.stdout.split("?")[1]
    # print("Iteration " + str(i) + " " + str(res))

    if res not in output:
        output['other'] += 1
    else:
        output[res] += 1

    if res not in faithful:
        faithful[res] = 0
    faithful[res] += 1

    if (i % 10 == 0):
        print("Output: ", output)
        print("faithful: ", faithful)

print(output)

try:
    f = open('log.txt', 'a')
    f.write('Model:{model}\n'.format(model=model))
    f.write('prompt:{p}\n'.format(p=prompt))
    f.write('Number of times experiment was conducted: k = {iter}\n'.format(iter=k))
    f.write('Temperature: {temp}'.format(temp=Temperature))
    f.write('result:\n')
    f.write(json.dumps(faithful))
    f.write("\n")
except TypeError:
    print("Type error occured\n")
except:
    print("Something else went wrong\n")

fig, ax = plt.subplots()
index = np.arange(len(ideal.keys()))
bar_width = 0.35

modelGen = plt.bar(index, output.values(), bar_width, color='red', label="Model generated")

idealDist = plt.bar(index + bar_width, ideal.values(), bar_width, color='g', label="Ideal probability distribution")

plt.xlabel("Strings in the grammar")
plt.ylabel("Frequency")
plt.title("Experiment: String of length {k}".format(k=k))
plt.xticks(index + bar_width, ideal.keys())
plt.legend()
plt.tight_layout()
plt.savefig("{grammar}{len}.pdf".format(grammar=grammar, len=k))

# result = {'1': 50, '01': 11, '001': 4, '0000000000000001': 3, '0001': 8, '0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000': 4, '000000000000000000000000000000000000000000000000000000000000000001': 2, '00000000001': 1, '000001': 4, '00001': 7, '00000000000000000000000000001': 1, '0000001': 1, '000000000001': 1, '000000000000000000000000000000000000000000000000000000000000001': 1, '000000001': 1, '000000000000000000000000000000000000001': 1}


# args = './main -m ./models/7B/ggml-model-q4_0.gguf -n 256 --grammar-file grammars/binary.gbnf --color'.split(' ')
# args = ['./main', '-m', './models/7B/ggml-model-q4_0.gguf', '-n', '256', '-p', 'Generate a binary string of length less then or equal to 3?', '--grammar-file', 'grammars/binary.gbnf',  '--color']