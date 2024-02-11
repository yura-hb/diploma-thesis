
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import List

from agents import Machine, WorkCenter


@dataclass
class Candidate:
    name: str
    work_center: WorkCenter
    machine: Machine


class Template(metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def from_cli(cls, parameters: dict) -> List['Candidate']:
        pass
