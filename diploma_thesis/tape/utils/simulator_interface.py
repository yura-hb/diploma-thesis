
from agents import MachineInput, WorkCenterInput
from agents.utils.memory import Record
from environment import ShopFloor, Machine, WorkCenter
from typing import TypeVar

MachineState = TypeVar('MachineState')
WorkCenterState = TypeVar('WorkCenterState')


class SimulatorInterface:

    def encode_machine_state(self, parameters: MachineInput) -> MachineState:
        pass

    def encode_work_center_state(self, parameters: WorkCenterInput) -> WorkCenterState:
        pass

    def did_prepare_machine_record(self,
                                   shop_floor: ShopFloor,
                                   machine: Machine,
                                   record: Record,
                                   decision_moment: int):
        pass

    def did_prepare_work_center_record(self,
                                       shop_floor: ShopFloor,
                                       work_center: WorkCenter,
                                       record: Record,
                                       decision_moment: int):
        pass
