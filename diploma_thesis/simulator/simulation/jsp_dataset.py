import os

from .simulation import Simulation


class JSPDataset:

    @classmethod
    def from_cli(cls, name: str, logger, parameters: dict) -> [Simulation]:
        path = parameters['path']
        simulations = []

        for file in os.listdir(path):
            if not file.endswith('.txt'):
                continue

            instance_name = file.split('.')[0]
            instance_path = os.path.join(path, file)

            with open(instance_path, 'r') as f:
                _, machines_count = f.readline().split()
                machines_count = int(machines_count)

            simulation = Simulation.from_cli(
                name=f'{name}-{instance_name}',
                logger=logger,
                parameters=dict(
                    configuration=dict(
                        timespan=100000,
                        machines_per_work_center=1,
                        work_center_count=machines_count,
                    ),
                    dispatch=dict(
                        job_sampler=dict(
                            kind='no',
                        ),
                        breakdown=dict(
                            kind='no'
                        ),
                        initial_job_assignment=dict(
                            kind='jsp_static',
                            parameters=dict(
                                path=instance_path
                            )
                        ),
                        seed=0
                    )
                )
            )

            simulations.append(simulation)

        return simulations
