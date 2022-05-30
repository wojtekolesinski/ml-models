import os
import time
from multiprocessing import Process, Manager


def find_best_vector(start, end, max, values, weights, number, return_dict):
    best_vector = 0
    best_value = 0
    best_weight = 0
    for current_vector in range(start, end):
        combined_weight = 0
        combined_value = 0

        for index, (value, weight) in enumerate(zip(values, weights)):
            if ((current_vector >> index) & 1) == 1:
                combined_weight += weight
                combined_value += value
                if combined_weight > max:
                    break

        if combined_weight <= max:
            if combined_value > best_value or (combined_value == best_value and combined_weight < best_weight):
                best_vector = current_vector
                best_value = combined_value
                best_weight = combined_weight
    return_dict[number] = (best_vector, best_value, best_weight)


if __name__ == '__main__':
    max_capacity = 50
    values = [1, 9, 5, 13, 3, 3, 6, 4, 11, 17, 7, 2, 3, 6, 1, 9, 7, 15, 1, 2, 3, 20, 5, 6, 12]
    weights = [11, 6, 2, 13, 4, 5, 7, 18, 9, 12, 2, 3, 15, 7, 5, 1, 2, 10, 8, 5, 17, 3, 7, 7, 2]

    max_vector = int('1' * len(values), 2)
    max_processes = os.cpu_count()
    processes = []
    bins = list(range(0, max_vector, max_vector // max_processes)) + [max_vector]

    manager = Manager()
    return_dict = manager.dict()

    for i, index in enumerate(range(len(bins) - 1)):
        p = Process(target=find_best_vector,
                    args=(bins[index], bins[index + 1], max_capacity, values, weights, i, return_dict))
        processes.append(p)
        p.start()

    start = time.time()

    for p in processes:
        p.join()

    print(f'elapsed time: {time.time() - start:.3}s')
    vector, value, weight = sorted(return_dict.values(), key=lambda x: x[1], reverse=True)[0]
    print(f'best vector: {bin(vector)[2:]} -> value: {value}, weight: {weight}')

    # best vector: 1001000111000011100000110 -> value: 112, weight: 49
    # best vector: 1001000111000011100000110 -> value: 112, weight: 49
