
import simpy

from dataclasses import dataclass, field
from .workflow import Workflow

from problem.problem import Problem
from problem.shopfloor import ShopFloor
from problem.work_center import WorkCenter
from problem.machine import Machine
from scheduling_rules.fifo import FIFOSchedulingRule
from scheduling_rules import SchedulingRule

from typing import Tuple, List
from functools import reduce


class Debug(Workflow):

    @dataclass
    class Configuration:
        environment: simpy.Environment = field(default_factory=simpy.Environment)

        problem: Problem = field(default_factory=Problem)

        scheduling_rule: SchedulingRule = field(default_factory=lambda: FIFOSchedulingRule())

    def __init__(self, configuration: Configuration = Configuration()):
        super().__init__()

        self.configuration = configuration

    def run(self):
        work_centers, machines_per_wc = self.__make_working_units__()
        machines = reduce(lambda x, y: x + y, machines_per_wc, [])

        shopfloor = ShopFloor(
            self.configuration.environment, machines, work_centers, self.configuration.problem
        )

        for work_center, machines_in_workcenter in zip(work_centers, machines_per_wc):
            work_center.connect(machines_in_workcenter, work_centers, self)

        for machine in machines:
            machine.connect(machines, work_centers, shopfloor)

        self.configuration.environment.run()

    def __make_working_units__(self) -> Tuple[List[WorkCenter], List[Machine]]:
        work_centers = []
        machines = []

        for work_center_idx in range(self.configuration.problem.workcenter_count):
            work_centers += [WorkCenter(self.configuration.environment, work_center_idx)]

            batch = []

            for machine_idx in range(self.configuration.problem.machines_per_workcenter):
                batch += [Machine(self.configuration.environment,
                                  machine_idx,
                                  work_center_idx,
                                  self.configuration.scheduling_rule)]

            machines += [batch]

        return work_centers, machines
