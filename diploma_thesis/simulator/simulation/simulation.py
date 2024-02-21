from dataclasses import dataclass
from logging import Logger
from typing import Dict

import simpy

import environment
from breakdown import from_cli as breakdown_from_cli
from environment import Agent, Delegate, ShopFloor
from job_sampler import from_cli as job_sampler_from_cli
from .cli import CLITemplate


class Simulation(CLITemplate):

    @dataclass
    class Configuration:
        configuration: Dict
        job_sampler: Dict
        breakdown: Dict

    def __init__(self, index: int, name: str, logger: Logger, configuration: Configuration):
        self.index = index
        self.name = name
        self.logger = logger
        self.shop_floor: ShopFloor = None
        self.configuration = configuration

    @property
    def simulation_id(self):
        return self.name

    def prepare(self, agent: Agent, delegate: Delegate, env: simpy.Environment):
        self.logger = self.logger.getChild(self.simulation_id)

        problem = self.configuration.configuration
        problem = environment.Configuration.from_cli_arguments(problem)

        sampler = job_sampler_from_cli(parameters=self.configuration.job_sampler, problem=problem, environment=env)
        breakdown = breakdown_from_cli(parameters=self.configuration.breakdown)

        configuration = ShopFloor.Configuration(
            problem=problem,
            sampler=sampler,
            agent=agent,
            delegate=delegate,
            environment=env,
            breakdown=breakdown
        )

        self.shop_floor = ShopFloor(self.index, self.simulation_id, configuration, self.logger)

    def run(self):
        yield self.shop_floor.simulate()

    def update_name(self, name: str):
        self.name = name

    def update_index(self, index: int):
        self.index = index

    @classmethod
    def from_cli(cls, name: str, logger: Logger, parameters: Dict):
        return cls(
            index=0,
            name=name,
            logger=logger,
            configuration=cls.Configuration(
                configuration=parameters['configuration'],
                job_sampler=parameters['job_sampler'],
                breakdown=parameters['breakdown']
            )
        )

