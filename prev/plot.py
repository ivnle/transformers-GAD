import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ['No Grammar', 'GCD: string_01.ebnf', 'GCD: string_start_w_1_all_0.ebnf',
           'GCD: string_start_w_1_all_0_explicit.ebnf', 'Additional Grammar: string_start_w_1_all_0.ebnf']
gen_counts = [845, 681, 500, 500, 1141]
g_star_counts = [500, 500, 500, 500, 500]
g_prime_counts = [258, 181, 0, 0, 187]
o_counts = [87, 0, 0, 0, 454]

x = np.arange(len(methods))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(14, 8))
rects1 = ax.bar(x - width*1.5, gen_counts, width, label='# gen')
rects2 = ax.bar(x - width/2, g_star_counts, width, label='# G*')
rects3 = ax.bar(x + width/2, g_prime_counts, width, label="# G'")
rects4 = ax.bar(x + width*1.5, o_counts, width, label='# O')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Counts')
ax.set_title('Counts by method and type')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45, ha="right")
ax.legend()

plt.tight_layout()
plt.show()


if __name__ == '__main__':

