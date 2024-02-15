from dataclasses import dataclass
from typing import Dict

import simpy

import environment
from environment import Agent, Delegate, ShopFloor
from job_sampler import from_cli as job_sampler_from_cli
from breakdown import from_cli as breakdown_from_cli
from .simulation import Simulation
from logging import Logger


class DynamicSimulation(Simulation):

    @dataclass
    class Configuration:
        configuration: Dict
        job_sampler: Dict
        breakdown: Dict

    def __init__(self, name: str, logger: Logger, configuration: Configuration):
        super().__init__(name, logger)

        self.configuration = configuration

    @property
    def simulation_id(self):
        return self.name

    def prepare(self, agent: Agent, delegate: Delegate, env: simpy.Environment):
        problem = self.configuration.configuration
        problem = environment.Configuration.from_cli_arguments(problem)

        sampler = job_sampler_from_cli(problem, env, self.configuration.job_sampler)
        breakdown = breakdown_from_cli(self.configuration.breakdown)

        configuration = ShopFloor.Configuration(
            problem=problem,
            sampler=sampler,
            agent=agent,
            delegate=delegate,
            environment=env,
            breakdown=breakdown
        )

        self.shop_floor = ShopFloor(self.simulation_id, configuration, self.logger)

    def run(self):
        yield self.shop_floor.simulate()

    @classmethod
    def from_cli(cls, name: str, logger: Logger, parameters: Dict):
        return cls(
            name=name,
            logger=logger,
            configuration=cls.Configuration(
                configuration=parameters['configuration'],
                job_sampler=parameters['job_sampler'],
                breakdown=parameters['breakdown']
            )
        )

