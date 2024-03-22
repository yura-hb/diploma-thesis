
import torch
import uuid

from .layer import *
from torch import nn


class Recurrent(Layer):

    def __init__(self, kind: str, memory_key: str, parameters: dict, signature: str):
        super().__init__(signature=signature)

        self.kind = kind
        self.memory_key = memory_key
        self.parameters = parameters
        self.model = None

        self.__build__()

    def forward(self, x, memory):
        match self.kind:
            case 'lstm':
                if memory is None:
                    output, (hidden_state, cell_state) = self.model(x)

                    return output, hidden_state, cell_state

                rnn_memory = memory[self.memory_key]

                hidden_state = rnn_memory['hidden_state'].unsqueeze(0)
                cell_state = rnn_memory['cell_state'].unsqueeze(0)

                if hidden_state.dim() == 2:
                    output, (hidden_state, cell_state) = self.model(x, (hidden_state, cell_state))

                    return output, hidden_state, cell_state

                x = x.unsqueeze(0)

                output, (hidden_state, cell_state) = self.model(x, (hidden_state, cell_state))

                return output.squeeze(0), hidden_state, cell_state
            case 'rnn' | 'gru':
                if memory is None:
                    return self.model(x)

                rnn_memory = memory[self.memory_key].unsqueeze(0)

                if rnn_memory.dim() == 2:
                    return self.model(x, rnn_memory)

                x = x.unsqueeze(0)

                x, hidden_state = self.model(x, rnn_memory)

                return x.squeeze(0), hidden_state
            case _:
                return None

    def __build__(self):
        match self.kind:
            case 'lstm':
                self.model = nn.LSTM(**self.parameters)
            case 'gru':
                self.model = nn.GRU(**self.parameters)
            case 'rnn':
                self.model = nn.RNN(**self.parameters)
            case _:
                raise ValueError(f'Unknown recurrent layer kind: {self.kind}')

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        kind = parameters['kind']
        memory_key = parameters['memory_key']
        signature = parameters['signature']

        del parameters['kind'], parameters['memory_key'], parameters['signature']

        return Recurrent(kind, memory_key=memory_key, parameters=parameters, signature=signature)
