import simpy

from abc import ABCMeta, abstractmethod
from environment import Agent, Delegate, ShopFloor
from logging import Logger


class Simulation(metaclass=ABCMeta):

    def __init__(self, name: str, logger: Logger):
        self.name = name
        self.logger = logger.getChild(f'Simulation {name}')
        self.shop_floor: ShopFloor = None

    @property
    @abstractmethod
    def simulation_id(self):
        return self.name

    @abstractmethod
    def prepare(self, agent: Agent, delegate: Delegate, env: simpy.Environment):
        pass

    @abstractmethod
    def run(self):
        pass
