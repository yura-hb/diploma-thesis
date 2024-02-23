
from abc import ABCMeta
from agents.base import Encoder
from typing import TypeVar
from agents.workcenter.utils import Input as WorkCenterInput

State = TypeVar('State')


class StateEncoder(Encoder[WorkCenterInput, State], metaclass=ABCMeta):

    Input = WorkCenterInput
