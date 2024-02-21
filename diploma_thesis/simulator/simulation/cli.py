
from abc import ABCMeta, abstractmethod
from typing import Dict
from logging import Logger
from flatdict import FlatDict
from itertools import product


class CLITemplate(metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def from_cli(cls, name: str, logger: Logger, parameters: Dict):
        pass
