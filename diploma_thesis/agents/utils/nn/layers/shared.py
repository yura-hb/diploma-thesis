import torch_geometric as pyg

from .layer import Layer
from .cli import from_cli


class Shared(Layer):

    def __init__(self, values: list[str], input_args, layers: [Layer]):
        values = ', '.join(values)
        signature = f'{values} -> {values}'

        super().__init__(signature=signature)
        #
        self.model = pyg.nn.Sequential(
            input_args,
            [(layer, layer.signature) for layer in layers]
        )

    def forward(self, *args) -> tuple:
        return tuple([self.model(arg) for arg in args])

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        return Shared(
            values=parameters['values'],
            input_args=parameters['input_args'],
            layers=[from_cli(layer) for layer in parameters['layers']]
        )
