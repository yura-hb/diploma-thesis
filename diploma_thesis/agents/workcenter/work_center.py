
from abc import ABCMeta

from agents.base.agent import Agent
from environment import WorkCenterKey
from .model import from_cli as model_from_cli
from .state import from_cli as state_encoder_from_cli


class WorkCenter(Agent[WorkCenterKey], metaclass=ABCMeta):
    pass
