
from abc import ABCMeta
from typing import TypeVar

from agents.base.agent import Agent
from .model import MachineModel
from .state import StateEncoder

MachineStateEncoder = TypeVar('MachineStateEncoder', bound=StateEncoder)
Model = TypeVar('Model', bound=MachineModel)


class Machine(Agent[MachineStateEncoder, Model], metaclass=ABCMeta):

    pass
