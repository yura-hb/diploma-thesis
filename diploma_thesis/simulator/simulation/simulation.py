from dataclasses import dataclass
from logging import Logger
from typing import Dict

import simpy

import environment

from environment import Agent, Delegate, ShopFloor
from dispatch import Dispatch


class Simulation:

    @dataclass
    class Configuration:
        configuration: Dict
        dispatch: Dict

    def __init__(self, index: int, name: str, logger: Logger, configuration: Configuration):
        self.index = index
        self.name = name
        self.logger = logger
        self.shop_floor: ShopFloor = None
        self.dispatch: Dispatch = None
        self.configuration = configuration

    @property
    def simulation_index(self):
        return self.index

    @property
    def simulation_id(self):
        return self.name

    def prepare(self, agent: Agent, delegate: Delegate, env: simpy.Environment):
        self.logger = self.logger.getChild(self.simulation_id)

        problem = self.configuration.configuration
        problem = environment.Configuration.from_cli_arguments(problem)

        configuration = ShopFloor.Configuration(
            configuration=problem,
            agent=agent,
            delegate=delegate,
            environment=env,
        )

        self.shop_floor = ShopFloor(self.index, self.simulation_id, configuration, self.logger)
        self.dispatch = Dispatch.from_cli(
            parameters=self.configuration.dispatch,
            problem=problem,
            environment=env
        )

    def run(self):
        yield self.dispatch.simulate(self.shop_floor)

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
                dispatch=parameters['dispatch']
            )
        )

