
import torch

from .layer import *


class Alias(Layer):

    def __init__(self, signature: str):
        super().__init__(signature=signature)

    def forward(self, *args):
        if len(args) == 1:
            return args[0]

        return args

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        return Alias(signature=parameters['signature'])
