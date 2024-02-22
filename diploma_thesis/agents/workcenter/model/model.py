from environment import Job, WaitInfo
from agents.base.model import *
from agents.workcenter import WorkCenterInput


class WorkCenterModel(Model[WorkCenterInput, State, Action, Job | WaitInfo], metaclass=ABCMeta):

    Input = WorkCenterInput


class NNWorkCenterModel(Model[NNModel, State, Action, Job | WaitInfo], metaclass=ABCMeta):

    Input = WorkCenterInput
