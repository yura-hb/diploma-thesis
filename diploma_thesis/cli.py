
import argparse
import yaml

from environment.configuration import Configuration
from workflows import SingleModel
from workflows import Workflow
from typing import Dict


def make_workflow(configuration: Dict) -> Workflow:
    match configuration['task']['id']:
        case "single_model":
            return SingleModel(configuration)
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
