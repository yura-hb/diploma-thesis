
import argparse

from environment.problem import Problem
from environment.job_samplers import add_cli_arguments, from_cli_arguments, JobSampler
from workflows import StaticRuleTournament
from workflows import StaticSingleRule
from workflows import Workflow


def make_workflow(id: str, problem: Problem, sampler: JobSampler) -> Workflow:
    match id:
        case "static_rule":
            return StaticSingleRule(problem, sampler=sampler)
        case "static_rule_tournament":
            return StaticRuleTournament(problem=problem)
        case _:
            raise ValueError(f"Unknown workflow id {id}")


def main(args: argparse.Namespace):  # pragma: no cover
    problem = Problem.from_cli_arguments(args)
    sampler = from_cli_arguments(problem, args)

    workflow = make_workflow(args.task, problem=problem, sampler=sampler)

    workflow.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", help="Run a workflow task")

    hidden = Problem.add_cli_arguments(parser)

    add_cli_arguments(hidden)

    args = parser.parse_args([] if "__file__" not in globals() else None)

    main(args)