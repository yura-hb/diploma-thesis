
from environment import Job
from agents.base.model import *
from agents.machine.utils import Input as MachineInput


class MachineModel(Model[MachineInput, Action, Job | None], metaclass=ABCMeta):

    Input = MachineInput


class DeepPolicyMachineModel(DeepPolicyModel[MachineInput, Action, Job | None], metaclass=ABCMeta):

    Input = MachineInput

    @classmethod
    def memory_key(cls, parameters: Input) -> None | str:
        return parameters.machine.shop_floor.id
