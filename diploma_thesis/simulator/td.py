
from .simulator import *


class TDSimulator(Simulator):
    """
    A simulator, which estimates returns in Temporal Difference manner and send information for training as soon as
    possible
    """

    def did_prepare_machine_record(self,
                                   shop_floor: ShopFloor,
                                   machine: Machine,
                                   record: Record,
                                   moment: float):
        self.machine.store(machine.key, record)

    def did_prepare_work_center_record(self,
                                       shop_floor: ShopFloor,
                                       work_center: WorkCenter,
                                       record: Record,
                                       moment: float):
        self.work_center.store(work_center.key, record)
