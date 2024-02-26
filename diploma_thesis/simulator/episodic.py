from .simulator import *


class EpisodicSimulator(Simulator):
    """
    A simulator, which launches several shop=floors in parallel and simulates them until terminating conditions are met.
    During the process of the run the whole episode of environment is recorded.

    After the run is finished returns are estimated and passed to the agent for training.
    """

    def __init__(self, machine: MachineAgent, work_center: WorkCenterAgent, tape_model: TapeModel):
        super().__init__(machine, work_center, tape_model)

        self.queue = dict()

    def did_prepare_machine_record(self, context: Context, machine: Machine, record: Record):
        super().did_prepare_machine_record(context, machine, record)

        pass

    def did_prepare_work_center_record(self, context: Context, work_center: WorkCenter, record: Record):
        super().did_prepare_work_center_record(context, work_center, record)

        pass

    @staticmethod
    def from_cli(parameters, *args, **kwargs) -> Simulator:
        return EpisodicSimulator(*args, **kwargs)
