from .agent import Agent
from .static import StaticCandidates
from .template import Template, Candidate

key_to_cls = {
    "agent": Agent,
    "static": StaticCandidates,
}


def from_cli(parameters: dict) -> list[Candidate]:
    cls = key_to_cls[parameters['kind']]

    return cls.from_cli(parameters['parameters'])
