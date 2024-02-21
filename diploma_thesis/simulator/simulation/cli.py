
from abc import ABCMeta, abstractmethod
from logging import Logger
from typing import Dict


class CLITemplate(metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def from_cli(cls, name: str, logger: Logger, parameters: Dict):
        pass
