
import torch
import numpy as np
import random
import argparse
from typing import Dict

import yaml

from workflow import Workflow, Simulation, Tournament, MultiSimulation

# torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
np.random.seed(0)


def make_workflow(configuration: Dict) -> Workflow:
    configuration = configuration['task']

    match configuration['kind']:
        case "task":
            return Simulation(configuration)
        case "multi_task":
            return MultiSimulation(configuration)
        case 'tournament':
            return Tournament(configuration)
        case _:
            raise ValueError(f"Unknown workflow id {id}")


def main(configuration):  # pragma: no cover
    workflow = make_workflow(configuration)

    workflow.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--configuration", help="Path to configuration file")

    args = parser.parse_args([] if "__file__" not in globals() else None)

    configuration_file = args.configuration

    with open(configuration_file) as file:
        configuration = yaml.safe_load(file)

    main(configuration)
