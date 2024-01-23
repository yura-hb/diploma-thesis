
import argparse
import simpy

from workflows import Debug
from workflows import Workflow
from environment.problem import Problem


def make_workflow(id: str, problem: Problem) -> Workflow:
    environment = simpy.Environment()

    match id:
        case "debug":
            configuration = Debug.Configuration(
                environment=environment,
                problem=problem
            )

            return Debug(configuration=configuration)
        case _:
            raise ValueError(f"Unknown workflow id {id}")


def main(args: argparse.Namespace):  # pragma: no cover
    problem = Problem.from_cli_arguments(args)
    workflow = make_workflow(args.task, problem=problem)

    workflow.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", help="Run a workflow task")

    Problem.add_cli_arguments(parser)

    args = parser.parse_args([] if "__file__" not in globals() else None)

    print(args)

    main(args)