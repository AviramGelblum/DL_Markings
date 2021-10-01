import math
from typing import Union


# region Subset Sum Solver Method
def find_approximate_subset(numbers: list[Union[float, int]], indices: list[int], goal: float, max_length: Union[int, float]) ->\
        tuple[list[Union[float, int]], list[int]]:
    """
    Heuristic algorithm for solving the subset sum problem, by sequentially selecting the closest number to a
    changing goal which is determined by the total subset sum, the maximum number of elements we allow in the output
    set, and how many numbers were already selected. Essentially, the more numbers are selected, the closer we allow
    the goal to be to the actual total subset sum goal.
    :param numbers: List of numbers defining the set of numbers from which the subset will be selected
    :param indices: List of indices for the numbers set. This is relevant when using this function for multi way
    partition.
    :param goal: Subset sum goal
    :param max_length: Maximum length allowed of each subset. Setting this to math.inf allows any subset length.
    :return: subset - Subset of numbers chosen from the given set of numbers to approximate the subset sum goal
             subset_indices - Indices of the numbers chosen in the original numbers list
    """

    # Validate max_length being integer or inf
    if isinstance(max_length, float):
        if max_length is not math.inf:
            raise ValueError('max_length can take on either an int or math.inf value')

    # Initialization
    subset, subset_indices = [], []
    sum_subset = sum(subset)
    if max_length == math.inf:
        # If there is no maximum set length, always select numbers which bring the subset sum as close as possible
        # to the goal (from below)
        divider = 1
    else:
        divider = max_length

    # Loop until subset sum is larger/equal to the goal, adding a single number in every iteration
    while sum_subset < goal:
        if len(subset) == max_length:
            # Maximum subset length reached, return current subset
            return subset, subset_indices

        # Calculate difference of each number (which wasn't already selected to the subset) from a transformed
        # goal value defined by the difference of the sum of the current subset from the subset sum goal, divided by
        # a linearly decreasing factor. This transformed goal value generates a better selection procedure which allows
        # sampling from the mid-sized numbers in the set, slowly converging towards the goal.
        differences: list = [(goal-sum_subset)/divider - i for i in numbers]

        # Select the closest number in the set
        if len(subset) == max_length-1:
            # If this is the last selection, select the closest number to the transformed goal value.
            diff_index = differences.index(min(differences, key=lambda y: abs(y)))
        else:
            try:
                # Select the closest number to the transformed goal value, conditioned on the new subset sum not
                # exceeding that transformed goal value.
                diff_index = differences.index(min([n for n in differences if n >= 0]))
            except ValueError:
                # If all of the numbers left are larger than the transformed goal value, select the closest number to
                # the transformed goal value.
                diff_index = differences.index(min(differences, key=lambda y: abs(y)))

        # Move the selected number and its index from the original lists to the subset lists.
        subset.append(numbers.pop(diff_index))
        subset_indices.append(indices.pop(diff_index))

        divider = max(divider - 1, 1)  # Update goal value divider
        sum_subset = sum(subset)  # Update subset sum
    return subset, subset_indices
# endregion


# region Balanced Multiway Partition Solver
def balanced_multi_way_partition(numbers: list[float], goal: float, max_length: [int, float], number_of_sets: int) ->\
        list[list[int]]:
    """
    Heuristic algorithm solving the balanced multi-way partition problem,to divide the original set of numbers
    to a given number of sets with the same number of elements (as much as possible), each summing to a
    number as close as possible to a given goal (the total sum divided by the given number of sets, usually). The
    algorithm achieves this by iterating over another algorithm which heuristically solves the subset sum problem
    with a given subset size constraint.
    :param numbers: List of numbers defining the set of numbers from which the subset will be selected
    :param goal: Subset sum goal for each subset
    :param max_length: Maximum length allowed of each subset. Setting this to math.inf allows any subset length.
    :param number_of_sets: Number of sets to divide the original set into, ina balanced fashion.
    :return: List of lists of indices of the numbers chosen in the original numbers list. Each list contains the
    indices of a single subset.
    """
    # Initialization
    indices = list(range(len(numbers)))
    subsets_indices = []
    i = 0
    while i in range(number_of_sets-1):
        # Find N-1 sets with subset sum equal to the goal, and constrained by size to be of maximum size given by
        # max_length.
        ss, subset_indices = find_approximate_subset(numbers, indices, goal, max_length)
        subsets_indices.append(subset_indices)  # Append list of indices of current subset to list of subset indices lists.
        # Because numbers and indices are mutable variables and each iteration of find_approximate_subset removes from
        # them the selected numbers (using the pop method), so the algorithm is successively applied on the correct
        # leftover set.
        print('subset {} sum is {}, subset {} length is {}'.format(i+1, sum(ss), i+1, len(ss)))
        i = i+1
    print('subset {} sum is {}, subset {} length is {}'.format(i+1, sum(numbers), i+1, len(numbers)))
    subsets_indices.append(indices)  # Append remaining numbers' indices as the last set. This is usually the least accurate set, but results are not bad.
    return subsets_indices
# endregion
