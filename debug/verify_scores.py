import math
import numpy as np

# scores = [16.291677474975586, 17.322086334228516]
scores = [11.404173851013184, 12.125460624694824]



# Calculate e^score for each score
exp_scores = [math.exp(score) for score in scores]

# Sum of all e^scores
sum_exp_scores = sum(exp_scores)

# Softmax for each score
softmax_scores = [exp_score / sum_exp_scores for exp_score in exp_scores]

print(softmax_scores)
print(math.log(0.6729))
