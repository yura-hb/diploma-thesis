
from agents.base.model import *
from agents.workcenter.utils import Input as WorkCenterInput
from environment import Machine


class WorkCenterModel(Model[WorkCenterInput, State, Action, Machine | None], metaclass=ABCMeta):

    Input = WorkCenterInput


class DeepPolicyWorkCenterModel(DeepPolicyModel[WorkCenterInput, State, Action, Machine | None], metaclass=ABCMeta):

    Input = WorkCenterInput
