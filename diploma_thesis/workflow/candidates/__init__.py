
from .template import Template, Candidate
from .agent import Agent
from .static import StaticCandidates

key_to_cls = {
    "agent": Agent,
    "static": StaticCandidates
}


def from_cli(parameters: dict) -> list[Candidate]:
    cls = key_to_cls[parameters['kind']]

    return cls.from_cli(parameters['parameters'])
