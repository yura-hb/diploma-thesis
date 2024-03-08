from abc import ABCMeta, abstractmethod
from typing import List, TypeVar

import torch

from torch_geometric.data import Batch

from agents.base import Encoder
from agents.machine import MachineInput

State = TypeVar('State')


class StateEncoder(Encoder[MachineInput, State], metaclass=ABCMeta):

    Input = MachineInput

    @staticmethod
    def __to_list_of_tensors__(parameters: List) -> List[torch.FloatTensor]:
        return [parameter if torch.is_tensor(parameter) else torch.tensor(parameter) for parameter in parameters]


class GraphStateEncoder(StateEncoder, metaclass=ABCMeta):

    def encode(self, parameters: StateEncoder.Input) -> State:
        result = self.__encode__(parameters)

        result.graph.data = Batch.from_data_list([result.graph.data])

        return result

    @abstractmethod
    def __encode__(self, parameters: StateEncoder.Input) -> State:
        pass
