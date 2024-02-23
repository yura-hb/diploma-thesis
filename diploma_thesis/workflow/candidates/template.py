import logging

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import List

from agents import MachineAgent, WorkCenterAgent


@dataclass
class Candidate:
    name: str
    machine: MachineAgent
    work_center: WorkCenterAgent


class Template(metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def from_cli(cls, parameters: dict) -> List['Candidate']:
        pass
