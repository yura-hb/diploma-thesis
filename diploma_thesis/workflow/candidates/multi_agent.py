from typing import Dict, List

from agents import machine_from_cli, work_center_from_cli
from .template import Template, Candidate
from utils.multi_value_cli import multi_value_cli


class MultiAgent(Template):

    @classmethod
    def from_cli(cls, parameters: Dict) -> List[Candidate]:
        return multi_value_cli(parameters,
                               lambda _params: Candidate(_params['name'],
                                                         work_center_from_cli(_params['work_center_agent']),
                                                         machine_from_cli(_params['machine_agent'])))

