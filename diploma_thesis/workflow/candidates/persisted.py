import os
from typing import Dict, List

from utils.persistence import load
from .template import Template, Candidate


class PersistedAgent(Template):

    @classmethod
    def from_cli(cls, parameters: Dict) -> List[Candidate]:
        path = parameters['path']
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

            try:
                machine = load(machine_file)
                work_center = load(work_center_file)

                result += [Candidate(file, machine=machine, work_center=work_center)]
            except:
                pass

        return result

