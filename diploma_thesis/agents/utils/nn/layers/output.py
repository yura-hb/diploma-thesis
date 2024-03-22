
from typing import List

import torch
from tensordict import TensorDict

from .keys import Key
from .layer import *


class Output(Layer):

    def __init__(self, keys: dict):
        leaves, signature = Output.make_signature(keys)

        super(Output, self).__init__(signature)

        self.keys = keys
        self.leaves = leaves

    def forward(self, *args):
        leaf_to_arg = dict(zip(self.leaves, args))

        result = self.__extract_values__(self.keys, leaf_to_arg)

        return result

    def __extract_values__(self, keys, leaf_to_arg: dict[str, torch.Tensor]):
        result = TensorDict({}, batch_size=[])

        for key, nested in keys.items():
            if isinstance(nested, dict):
                result[key] = self.__extract_values__(nested, leaf_to_arg)
            else:
                result[key] = leaf_to_arg[nested]

        return result

    @staticmethod
    def make_signature(keys: dict) -> tuple[List[str], str]:
        leaves = Output.leaf_values(keys)
        leaves = set(leaves)
        leaves = sorted(leaves)

        return leaves, f"{', '.join(leaves)} -> {Key.OUTPUT}"

    @staticmethod
    def leaf_values(keys: dict) -> List[str]:
        result = []

        if isinstance(keys, str):
            return [keys]

        if isinstance(keys, dict):
            for _, keys_ in keys.items():
                result.extend(Output.leaf_values(keys_))

        return result

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        return Output(keys=parameters)
