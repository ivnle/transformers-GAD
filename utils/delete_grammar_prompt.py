import sys

input_filename = sys.argv[1]
output_filename = sys.argv[2]

with open(input_filename, 'r') as file:
    content = file.read()

blocks = content.split('\n\n')

new_content = []

for block in blocks:
    lines = block.split('\n')
    if lines[0].startswith('(synth-fun'):
        new_content.append(lines[0].strip() + ')')
    else:
        new_content.append(block)

result = '\n\n'.join(new_content)

with open(output_filename, 'w') as file:
    file.write(result)