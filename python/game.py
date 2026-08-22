import random

#Odd cycle Cn, n ≥ 3. The referee draws one question uniformly from 2n.

random.seed(42) #for reproducibility

N = 3 #start with case n = 3

def question_order(n):
    return