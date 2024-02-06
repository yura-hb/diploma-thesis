from dataclasses import dataclass
from typing import Dict

import simpy

import environment
from environment import Agent, Delegate, ShopFloor
from job_samplers import from_cli_arguments as job_sampler_from_cli_arguments
from .simulation import Simulation


@dataclass
class Configuration:
    idx: int
    configuration: Dict
    sampler: Dict


class DynamicSimulation(Simulation):

    def __init__(self, configuration: Configuration):
        self.configuration = configuration

    def run(self, shop_floor_idx: int, agent: Agent, delegate: Delegate, env: simpy.Environment):
        problem = self.configuration.configuration
        problem = environment.Configuration.from_cli_arguments(problem)

        sampler = job_sampler_from_cli_arguments(problem, env, self.configuration.sampler)

        configuration = ShopFloor.Configuration(
            problem=problem, sampler=sampler, agent=agent, delegate=delegate, environment=env
        )

        shop_floor = ShopFloor(shop_floor_idx, configuration, None)

        shop_floor.simulate()
