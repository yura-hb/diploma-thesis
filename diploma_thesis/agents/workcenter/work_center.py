
from abc import ABCMeta

from agents.base.agent import Agent
from environment import WorkCenterKey


class WorkCenter(Agent[WorkCenterKey], metaclass=ABCMeta):
    pass
