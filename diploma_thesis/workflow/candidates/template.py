from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import List

import agents
from agents.machine import StaticMachine
from agents.machine.model import StaticMachineModel
from agents.machine.state import PlainEncoder
from agents.workcenter import StaticWorkCenter
from agents.workcenter.model import StaticWorkCenterModel
from agents.workcenter.state import PlainEncoder
from utils.persistence import load


@dataclass
class Candidate:
    name: str
    kind: str
    parameters: dict

    def load(self):
        match self.kind:
            case 'static':
                scheduling_rule = self.parameters['scheduling_rule']
                routing_rule = self.parameters['routing_rule']

                return StaticMachine(
                    model=StaticMachineModel(scheduling_rule),
                    state_encoder=PlainEncoder()
                ), StaticWorkCenter(
                    model=StaticWorkCenterModel(routing_rule),
                    state_encoder=PlainEncoder()
                )
            case 'agent':
                machine = agents.machine_from_cli(self.parameters['machine_agent'])
                work_center = agents.work_center_from_cli(self.parameters['work_center_agent'])

                machine.load_state_dict(load(self.parameters['machine_file']))
                work_center.load_state_dict(load(self.parameters['work_center_file']))

                return machine, work_center


class Template(metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def from_cli(cls, parameters: dict) -> List['Candidate']:
        pass
