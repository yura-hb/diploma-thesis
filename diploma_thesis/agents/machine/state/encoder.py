from abc import ABCMeta
from typing import List, TypeVar

import torch

from agents.base import Encoder
from agents.machine import MachineInput

State = TypeVar('State')


class StateEncoder(Encoder[MachineInput, State], metaclass=ABCMeta):

    Input = MachineInput

    @staticmethod
    def __to_list_of_tensors__(parameters: List) -> List[torch.FloatTensor]:
        return [parameter if torch.is_tensor(parameter) else torch.tensor(parameter) for parameter in parameters]
