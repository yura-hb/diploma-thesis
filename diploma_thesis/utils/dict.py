
import copy

from typing import Dict
from flatdict import FlatDict
from itertools import product


FACTORY_SUFFIX = '__factory__'


def merge_dicts(lhs: Dict, rhs: Dict) -> Dict:
    """
    Recursively merge two dictionaries
    """

    result = copy.deepcopy(lhs)

    for key, value in rhs.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        elif key in result and isinstance(result[key], list) and isinstance(value, list):
            lhs = result[key]
            rhs = value

            # Merge existing elements
            for i in range(min(len(lhs), len(rhs))):
                if isinstance(lhs[i], dict) and isinstance(rhs[i], dict):
                    lhs[i] = merge_dicts(lhs[i], rhs[i])
                else:
                    lhs[i] = rhs[i]

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
        if key.endswith(FACTORY_SUFFIX):
            flatten[key.rstrip("." + FACTORY_SUFFIX)] = __parse_factory__(value)

            continue

        if not isinstance(value, list):
            flatten[key] = [value]

    keys = flatten.keys()
    values = flatten.values()

    for combination in product(*values):
        result = FlatDict(zip(keys, combination), delimiter=delimiter)

        yield result.as_dict()


def __parse_factory__(parameters):
    result = []

    for combination in product(*parameters):
        values = []

        for parameter in combination:
            if isinstance(parameter, list):
                values.extend(parameter)
            else:
                values.append(parameter)

        result.append(values)

    return result