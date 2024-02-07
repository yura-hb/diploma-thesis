
from environment import Job, WaitInfo
from agents.base.model import *
from agents.machine import MachineInput


class MachineModel(Model[MachineInput, State, Action, Job | WaitInfo], metaclass=ABCMeta):

    Input = MachineInput
