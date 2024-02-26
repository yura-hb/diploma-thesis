from .simulator import *


class TDSimulator(Simulator):
    """
    A simulator, which estimates returns in Temporal Difference manner and send information for training as soon as
    possible
    """

    def did_prepare_machine_record(self, context: Context, machine: Machine, record: Record):
        super().did_prepare_machine_record(context, machine, record)

        self.machine.store(machine.key, record)

    def did_prepare_work_center_record(self, context: Context, work_center: WorkCenter, record: Record):
        super().did_prepare_work_center_record(context, work_center, record)

        self.work_center.store(work_center.key, record)

    @staticmethod
    def from_cli(parameters, *args, **kwargs) -> Simulator:
        return TDSimulator(*args, **kwargs)
