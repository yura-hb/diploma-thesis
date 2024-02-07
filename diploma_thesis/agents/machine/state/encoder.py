
from abc import ABCMeta
from agents.base import Encoder
from typing import TypeVar
from agents.machine import MachineInput

State = TypeVar('State')


class StateEncoder(Encoder[MachineInput, State], metaclass=ABCMeta):

    Input = MachineInput
