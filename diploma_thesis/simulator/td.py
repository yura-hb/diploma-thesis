from agents.base.agent import Slice, Trajectory
from .simulator import *
from .utils import TDQueue


receives = 0

class TDSimulator(Simulator):
    """
    A simulator, which estimates returns in Temporal Difference manner and send information for training as soon as
    possible
    """

    def __init__(self,
                 memory: int = 1,
                 emit_trajectory: bool = False,
                 reset_trajectory: bool = True,
                 sliding_window: int = 16,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.memory = memory
        self.emit_trajectory = emit_trajectory
        self.reset_trajectory = reset_trajectory
        self.sliding_window = sliding_window
        self.episode = 0

        self.machine_queue = TDQueue(self.machine.is_distributed)
        self.work_center_queue = TDQueue(self.work_center.is_distributed)

    def schedule(self, context: Context, machine: Machine) -> Job | None:
        job = super().schedule(context, machine)

        self.machine_queue.reserve(context.shop_floor.id, machine.key, context.moment)

        return job

    def route(self, context: Context, work_center: WorkCenter, job: Job) -> 'Machine | None':
        job = super().route(context, work_center, job)

        self.work_center_queue.reserve(context.shop_floor.id, work_center.key, context.moment)

        return job

    def did_prepare_machine_record(self, context: Context, machine: Machine, record: Record):
        super().did_prepare_machine_record(context, machine, record)

        # self.machine.store(machine.key, Slice(episode_id=context.shop_floor.id, records=[record]))

        if self.memory == 1:
            self.machine.store(machine.key, Slice(episode_id=context.shop_floor.id, records=[record]))
            return

        global receives

        receives += 1

        print(f'Total Receives: {receives}')

        self.machine_queue.store(context.shop_floor.id, machine.key, context.moment, record)

        self.__forward_td__(context, self.machine_queue, self.machine, machine.key)

    def did_prepare_work_center_record(self, context: Context, work_center: WorkCenter, record: Record):
        super().did_prepare_work_center_record(context, work_center, record)

        self.work_center_queue.store(context.shop_floor.id, work_center.key, context.moment, record)

        self.__forward_td__(context, self.work_center_queue, self.work_center, work_center.key)

    def __forward_td__(self, context: Context, queue: TDQueue, agent, key):
        prefix = queue.prefix(context.shop_floor.id, key, self.memory)

        if prefix is None:
            return

        moments, records = list(zip(*prefix))

        if self.emit_trajectory:
            agent.store(key, Trajectory(episode_id=self.episode, records=records))
            self.episode += 1
        else:
            agent.store(key, Slice(episode_id=context.shop_floor.id, records=records))

        suffix = max(1, self.sliding_window)

        if (not self.emit_trajectory or not self.reset_trajectory) and len(prefix) > suffix:
            data = list(zip(moments, records))[suffix:]

            queue.store_slice(context.shop_floor.id, key, dict(data))

    @staticmethod
    def from_cli(parameters, *args, **kwargs) -> Simulator:
        return TDSimulator(parameters.get('memory', 1), 
                           parameters.get('emit_trajectory', False),
                           parameters.get('reset_trajectory', True),
                           parameters.get('sliding_window', 1),
                           *args, **kwargs)