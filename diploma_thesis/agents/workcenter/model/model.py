from environment import Job, Machine
from agents.base.model import *
from agents.workcenter.utils import Input as WorkCenterInput


class WorkCenterModel(Model[WorkCenterInput, State, Action, Machine | None], metaclass=ABCMeta):

    Input = WorkCenterInput


class NNWorkCenterModel(Model[NNModel, State, Action, Machine | None], metaclass=ABCMeta):

    Input = WorkCenterInput
