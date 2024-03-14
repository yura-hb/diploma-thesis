import os
import yaml
from typing import Dict, List

import agents
from utils.persistence import load
from .template import Template, Candidate


class PersistedAgent(Template):

    @classmethod
    def from_cli(cls, parameters: Dict) -> List[Candidate]:
        path = parameters['path']
        prefix = parameters.get('prefix', '')
        result = []

        for file in os.listdir(path):
            target_dir = os.path.join(path, file)

            if not os.path.isdir(target_dir):
                continue

            agents_dir = os.path.join(path, file, 'agent')
            machine_file = os.path.join(agents_dir, 'machine.pt')
            work_center_file = os.path.join(agents_dir, 'work_center.pt')

            if not os.path.exists(machine_file) or not os.path.exists(work_center_file):
                continue

            parameters = os.path.join(target_dir, 'parameters.yml')

            with open(parameters, 'r') as f:
                parameters = yaml.load(f, Loader=yaml.FullLoader)

            try:
                machine = agents.machine_from_cli(parameters['machine_agent'])
                work_center = agents.work_center_from_cli(parameters['work_center_agent'])

                machine.load_state_dict(load(machine_file))
                work_center.load_state_dict(load(work_center_file))

                result += [Candidate(prefix + '_' + file,
                                     parameters=parameters,
                                     machine=machine,
                                     work_center=work_center)]
            except:
                pass

        return result

