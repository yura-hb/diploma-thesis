

from typing import List
from .dict import merge_dicts, iterate_all_combinations


def multi_value_cli(parameters: dict, build_fn) -> List:
    base_parameters = parameters['base']
    values = parameters['values']

    result = []

    for combination in iterate_all_combinations(values):
        tmp = merge_dicts(base_parameters, combination)

        result += [build_fn(tmp)]

    return result
