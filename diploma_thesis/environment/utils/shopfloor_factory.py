
import environment
from typing import List, Tuple
from functools import reduce


class ShopFloorFactory:

    def __init__(self, configuration: environment.ShopFloor.Configuration):
        self.configuration = configuration

    def make(self):
        work_centers, machines_per_wc = self.__make_working_units__()

        machines = reduce(lambda x, y: x + y, machines_per_wc, [])

        for work_center, machines_in_work_center in zip(work_centers, machines_per_wc):
            work_center.connect(machines_in_work_center, work_centers, self)

        for machine in machines:
            machine.connect(machines, work_centers, self)

        return work_centers, machines

    def __make_working_units__(self) -> Tuple[List[environment.WorkCenter], List[environment.Machine]]:
        work_centers = []
        machines = []

        for work_center_idx in range(self.configuration.problem.workcenter_count):
            work_centers += [environment.WorkCenter(self.configuration.environment,
                                                    work_center_idx,
                                                    rule=self.configuration.routing_model)]

            batch = []

            for machine_idx in range(self.configuration.problem.machines_per_workcenter):
                batch += [environment.Machine(self.configuration.environment,
                                              machine_idx,
                                              work_center_idx,
                                              self.configuration.scheduling_model)]

            machines += [batch]

        return work_centers, machines
