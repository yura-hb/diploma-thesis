import logging

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import List

from agents import Machine, WorkCenter


@dataclass
class Candidate:
    name: str
    machine: Machine
    work_center: WorkCenter


class Template(metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def from_cli(cls, parameters: dict) -> List['Candidate']:
        pass
