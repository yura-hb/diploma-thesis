
from agents.base.agent import Agent
from agents.machine import from_cli as machine_from_cli
from agents.machine.utils import Input as MachineInput
from agents.workcenter import from_cli as work_center_from_cli
from agents.workcenter.utils import Input as WorkCenterInput
from environment import MachineKey, WorkCenterKey
from .machine import StaticMachine
from .utils.phase import *

MachineAgent = Agent[MachineKey]
WorkCenterAgent = Agent[WorkCenterKey]


