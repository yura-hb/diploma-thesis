
from typing import Dict
from flatdict import FlatDict
from itertools import product


def merge_dicts(lhs: Dict, rhs: Dict) -> Dict:
    """
    Recursively merge two dictionaries
    """

    result = lhs.copy()

    for key, value in rhs.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        elif key in result and isinstance(result[key], list) and isinstance(value, list):
            lhs = result[key]
            rhs = value

            # Merge existing elements
            for i in range(min(len(lhs), len(rhs))):
                lhs[i] = merge_dicts(lhs[i], rhs[i])

            # Pad with not presented elements
            if len(lhs) < len(rhs):
                lhs += rhs[len(lhs):]

            result[key] = lhs
        else:
            result[key] = value

    return result


def iterate_all_combinations(value: Dict) -> [Dict]:
    """
    A function which receives a dictionary, where each value is either a list or a single value, and yield all
    possible combinations of list values.
    """
    delimiter = '.'
    flatten = FlatDict(value, delimiter=delimiter)

    for key, value in flatten.items():
        if not isinstance(value, list):
            flatten[key] = [value]

    keys = flatten.keys()
    values = flatten.values()

    for combination in product(*values):
        result = FlatDict(zip(keys, combination), delimiter=delimiter)

        yield result.as_dict()
