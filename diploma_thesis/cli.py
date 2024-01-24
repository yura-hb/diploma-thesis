
import argparse

import simpy

from environment.problem import Problem
from environment.shopfloor import ShopFloor

from workflows import StaticRuleTournament
from workflows import StaticSingleRule
from workflows import Workflow


def make_workflow(id: str, problem: Problem) -> Workflow:
    match id:
        case "static_rule":
            return StaticSingleRule(problem)
        case "static_rule_tournament":
            return StaticRuleTournament(problem=problem)
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