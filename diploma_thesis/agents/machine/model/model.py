
from environment import Job
from agents.base.model import *
from agents.machine.utils import Input as MachineInput


class MachineModel(Model[MachineInput, State, Action, Job | None], metaclass=ABCMeta):

    Input = MachineInput


class NNMachineModel(Model[MachineInput, State, Action, Job | None], metaclass=ABCMeta):

    Input = MachineInput
