
import environment
import weakref
from typing import List, Tuple
from functools import reduce


class ShopFloorFactory:

    def __init__(self, configuration: 'environment.ShopFloor.Configuration', shop_floor: 'environment.ShopFloor'):
        self.configuration = configuration
        self.shop_floor = weakref.ref(shop_floor)

    def make(self):
        result = self.__make_working_units__()
        work_centers = result[0]
        machines_per_wc = result[1]

        machines: List[environment.Machine] = reduce(lambda x, y: x + y, machines_per_wc, [])

        for work_center, machines_in_work_center in zip(work_centers, machines_per_wc):
            work_center.connect(shop_floor=self.shop_floor, machines=machines_in_work_center)

        for machine in machines:
            machine.connect(self.shop_floor)

        return work_centers, machines

    def __make_working_units__(self) -> Tuple[List[environment.WorkCenter], List[environment.Machine]]:
        work_centers = []
        machines = []

        for work_center_idx in range(self.configuration.configuration.work_center_count):
            work_centers += [environment.WorkCenter(self.configuration.environment, work_center_idx)]

            batch = []

            for machine_idx in range(self.configuration.configuration.machines_per_work_center):
                batch += [environment.Machine(self.configuration.environment, machine_idx, work_center_idx)]

            machines += [batch]

        return work_centers, machines
