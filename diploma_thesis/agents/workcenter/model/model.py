
from agents.base.model import *
from agents.workcenter.utils import Input as WorkCenterInput
from environment import Machine


class WorkCenterModel(Model[WorkCenterInput, Action, Machine | None], metaclass=ABCMeta):

    Input = WorkCenterInput


class DeepPolicyWorkCenterModel(DeepPolicyModel[WorkCenterInput, Action, Machine | None], metaclass=ABCMeta):

    Input = WorkCenterInput

    @classmethod
    def memory_key(cls, parameters: Input) -> None | str:
        return parameters.work_center.shop_floor.id
