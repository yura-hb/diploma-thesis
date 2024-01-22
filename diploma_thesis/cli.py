
import argparse

from .workflows import Debug


def make_workflow(id: str):
    match id:
        case "debug":
            configuration = DebugConfiguration()

            return Debug()
        case _:
            raise ValueError(f"Unknown workflow id {id}")


def main(args: argparse.Namespace):  # pragma: no cover
    workflow = make_workflow(args.task)

    workflow.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", help="Run a workflow task")

    sub_parsers = parser.add_subparsers(dest="problem")
    sub_parser = sub_parsers.add_parser(id="problem", parents=[parser])

    sub_parser.add_argument("--bla", help="Run a workflow task")

    args = parser.parse_args([] if "__file__" not in globals() else None)

    print(args)

    main(args)