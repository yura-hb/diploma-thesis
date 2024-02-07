
from abc import ABCMeta
from typing import TypeVar

from agents.base.agent import Agent
from .model import WorkCenterModel
from .state import StateEncoder

WorkCenterStateEncoder = TypeVar('WorkCenterStateEncoder', bound=StateEncoder)
Model = TypeVar('Model', bound=WorkCenterModel)


class WorkCenter(Agent[WorkCenterStateEncoder, Model], metaclass=ABCMeta):

    pass
