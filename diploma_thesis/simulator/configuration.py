from dataclasses import dataclass, field
from functools import partial
from logging import Logger
from typing import List, Dict

from utils import from_cli
from .simulation import Simulation, from_cli as simulation_from_cli


@dataclass
class RunConfiguration:
    @dataclass
    class TimelineSchedule:
        # List of durations for warm up phases
        warm_up_phases: List[float] = field(default_factory=list)
        # Maximum Duration of run
        duration: int = 1000

        @staticmethod
        def from_cli(parameters: Dict):
            return RunConfiguration.TimelineSchedule(
                warm_up_phases=parameters['warmup'],
                duration=parameters['duration']
            )

    @dataclass
    class TrainSchedule:
        pretrain_steps: int = 0
        train_interval: int = 0
        max_training_steps: int = 0

        @staticmethod
        def from_cli(parameters: Dict):
            return RunConfiguration.TrainSchedule(
                pretrain_steps=parameters['pretrain_steps'],
                train_interval=parameters['train_interval'],
                max_training_steps=parameters['max_training_steps']
            )

    timeline: TimelineSchedule
    machine_train_schedule: TrainSchedule
    work_center_train_schedule: TrainSchedule
    n_workers: int = 4
    simulations: List[Simulation] = field(default_factory=list)

    @classmethod
    def from_cli(cls, parameters: Dict, logger: Logger):
        return cls(
            timeline=cls.TimelineSchedule.from_cli(parameters['timeline']),
            machine_train_schedule=cls.TrainSchedule.from_cli(parameters['machine_train_schedule']),
            work_center_train_schedule=cls.TrainSchedule.from_cli(parameters['work_center_train_schedule']),
            n_workers=parameters['n_workers'],
            simulations=simulation_from_cli(prefix=parameters.get('prefix', ''),
                                            parameters=parameters['simulations'],
                                            logger=logger)
        )


@dataclass
class EvaluateConfiguration:
    n_workers: int = 4
    simulations: List[Simulation] = field(default_factory=list)

    @classmethod
    def from_cli(cls, logger: Logger, parameters: Dict):
        return cls(
            n_workers=parameters['n_workers'],
            simulations=simulation_from_cli(prefix=parameters.get('prefix', ''),
                                            parameters=parameters['simulations'],
                                            logger=logger)
        )


run_configuration_to_cls = {
    'plain': RunConfiguration,
}

run_configuration_from_cli = partial(from_cli, key_to_class=run_configuration_to_cls)

eval_configuration_to_cls = {
    'plain': EvaluateConfiguration,
}

evaluate_configuration_from_cli = partial(from_cli, key_to_class=eval_configuration_to_cls)
