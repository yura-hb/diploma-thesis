from typing import TypeVar

from agents.utils.memory import Record
from environment import ShopFloor, Machine, WorkCenter, Job, Context

MachineState = TypeVar('MachineState')
WorkCenterState = TypeVar('WorkCenterState')


class SimulatorInterface:

    def encode_machine_state(self, context: Context, machine: Machine, memory) -> MachineState:
        pass

    def encode_work_center_state(self, context: Context, work_center: WorkCenter, job: Job, memory) -> WorkCenterState:
        pass

    def did_prepare_machine_record(self, context: Context, machine: Machine, record: Record):
        pass

    def did_prepare_work_center_record(self, context: Context, work_center: WorkCenter, record: Record):
        pass
