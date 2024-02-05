
from .scheduling_model import SchedulingModel
from agent.scheduling.static.static_scheduling_model import StaticSchedulingModel
from typing import Dict

key_to_model = {
    'static': StaticSchedulingModel,
}


def from_cli_arguments(configuration: Dict) -> SchedulingModel:
    model = key_to_model[configuration['id']]

    return model(**configuration['parameters'])