from abc import ABCMeta

from agents.base.agent import Agent

from .model import MachineModel, from_cli as model_from_cli
from .state import StateEncoder, from_cli as state_encoder_from_cli


class Machine(Agent, metaclass=ABCMeta):
    pass
