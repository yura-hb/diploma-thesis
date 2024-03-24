import os
from typing import Dict, List

import yaml

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

            parameters['machine_file'] = machine_file
            parameters['work_center_file'] = work_center_file

            result += [Candidate(prefix + '_' + file,
                                 kind='agent',
                                 parameters=parameters)]

        return result

