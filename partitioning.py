import numpy as np
import copy


def find_approximate_subset(numbers_in, indices, goal, max_length=None, pop=False):
    if pop:
        numbers = numbers_in
    else:
        numbers = copy.deepcopy(numbers_in)

    if max_length is None:
        max_length = max(round(goal/np.median(numbers)), 2)  # probably we can set this in a smarter way

    if type(numbers) == np.ndarray:
        numbers = list(numbers)

    subset = []
    subset_indices = []
    sum_subset = sum(subset)
    multiplier = max_length

    while sum_subset < goal:
        if len(subset) == max_length:
            return subset, subset_indices
        differences = [((goal-sum_subset)/multiplier) - i for i in numbers]
        multiplier = max(multiplier-1, 1)
        if len(subset) == max_length-1:
            diff_index = differences.index(min(differences, key=lambda y: abs(y)))
        else:
            try:
                diff_index = differences.index(min([n for n in differences if n >= 0]))
            except ValueError:  # case where all of the numbers are larger than the goal
                diff_index = differences.index(min(differences, key=lambda y: abs(y)))

        subset.append(numbers.pop(diff_index))
        subset_indices.append(indices.pop(diff_index))
        sum_subset = sum(subset)
    return subset, subset_indices


def balanced_multi_way_partition(numbers: list, goal, max_length, number_of_sets):
    indices = list(range(len(numbers)))
    subsets_indices = []
    for _ in range(number_of_sets-1):
        _, subset_indices = find_approximate_subset(numbers, indices, goal, max_length, pop=True)
        subsets_indices.append(subset_indices)
    #   print('subset {} sum is {}, subset {} length is {}'.format(i+1, sum(ss), i+1, len(ss)))
    # print('subset {} sum is {}, subset {} length is {}'.format(i+2, sum(numbers), i+2, len(numbers)))
    subsets_indices.append(indices)
    return subsets_indices

if __name__ == '__main__':
    import itertools
    # for goal, goal_length in itertools.product(range(1, 25), range(1,5)):
    #     numbers = [2, 1, 4, 12, 15, 3, 7]
    #     subset = find_approximate_subset(numbers, goal, goal_length)
    #     print('goal is {}, length is {}, subset is {},\n difference between actual and found is {}'.format(str(goal),
    #           str(goal_length), str(subset), str(abs(sum(subset)-goal))))
    # for goal in range(1, 25):
    #     numbers = [2, 1, 4, 12, 15, 3, 7, 22, 31, 1 , 1 , 1, 3, 4 , 5 ,6 ,7 ,3, 2 , 8]
    #     subset = find_approximate_subset(numbers, goal)
    #     print('goal is {}, length is {}, subset is {},\n difference between actual and found is {}'.format(str(goal),
    #           str(subset.__len__()), str(subset), str(abs(sum(subset)-goal))))
