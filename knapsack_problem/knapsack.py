import time
import numpy as np
from multiprocessing import process

with open('data/plecak.txt', 'r') as data:
    max_capacity, n = [int(v) for v in data.readline().split(' ')]
    values = np.array([int(v) for v in data.readline().split(',')])
    weights = np.array([int(v) for v in data.readline().split(',')])

start = time.time()
best_vector = 0
best_value = 0
best_weight = max_capacity
max_vector = int('1' * len(values), 2)
current_vector = max_vector
for current_vector in range(max_vector, 0, -1):
    combined_weight = 0
    combined_value = 0

    # str_vec = bin(current_vector)[2:]

    # if len(str_vec) < len(values):
    #     str_vec = ('0' * (len(values) - len(str_vec))) + str_vec
    # filter_vector = np.array([int(x) for x in str_vec])
    #
    # # print(filter_vector)
    # combined_value = np.sum(values * filter_vector)
    # combined_weight = np.sum(weights * filter_vector)

    for index, (value, weight) in enumerate(zip(values, weights)):
        if ((current_vector >> index) & 1) == 1:
            combined_weight += weight
            combined_value += value
            if combined_weight > max_capacity:
                break

    if combined_weight <= max_capacity:
        if combined_value > best_value or (combined_value == best_value and combined_weight < best_weight):
            best_vector = current_vector
            best_value = combined_value
            best_weight = combined_weight
            print(f'{(max_vector-current_vector) / max_vector:.2%}',
                  bin(best_vector)[2:],
                  best_value,
                  best_weight,
                  sep='\t')

print(f'elapsed time: {time.time() - start:.3}s')
