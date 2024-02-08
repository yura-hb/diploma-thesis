import simpy

from abc import ABCMeta, abstractmethod
from environment import Agent, Delegate, ShopFloor
from logging import Logger


class Simulation(metaclass=ABCMeta):

    def __init__(self, name: str, logger: Logger):
        self.name = name
        self.logger = logger.getChild(f'simulation {name}')
        self.shop_floor: ShopFloor = None

    @property
    @abstractmethod
    def shop_floor_id(self):
        pass

    def run(self, agent: Agent, delegate: Delegate, env: simpy.Environment):
        pass
