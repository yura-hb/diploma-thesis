
import simpy

from abc import ABCMeta
from environment import Delegate, Agent
from agents.machine import Agent as Model


class Simulator(Agent, Delegate, metaclass=ABCMeta):

    def __init__(self, agent: Model):
        self.agent = agent
        self.environment = simpy.Environment()

    def simulate(self):
        pass