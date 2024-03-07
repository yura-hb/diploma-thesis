from typing import Dict

from agents import machine_from_cli, work_center_from_cli
from .template import Template, Candidate


class Agent(Template):

    @classmethod
    def from_cli(cls, parameters: Dict):
        name = parameters['name']
        machine = machine_from_cli(parameters['machine_agent'])
        work_center = work_center_from_cli(parameters['work_center_agent'])

        return [Candidate(name, parameters, work_center, machine)]
