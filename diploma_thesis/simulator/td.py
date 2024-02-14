from typing import List, TypeVar

from agents import MachineInput, WorkCenterInput
from environment import Job, WorkCenter, Machine, WaitInfo
from .simulator import Simulator

State = TypeVar('State')
Action = TypeVar('Action')


class TDSimulator(Simulator):
    """
    A simulator, which estimates returns in Temporal Difference manner and send information for training as soon as
    possible
    """

    # @dataclass(frozen=True)
    # class Key:
    #     shop_floor_id: str
    #     work_center_id: int
    #     machine_id: int
    #
    # @dataclass
    # class Record:
    #     state: State
    #     action: Action
    #     next_state: State
    #     reward: torch.FloatTensor
    #     done: bool
    #
    # def __post_init__(self):
    #     self.queue: Dict[TDSimulator.Key, TDSimulator.Record] = {}

    def schedule(self, shop_floor_id: str, machine: Machine, now: int) -> Job | WaitInfo:
        parameters = MachineInput(machine, now)
        result = self.machine.schedule(parameters)

        # self.queue[TDSimulator.Key(shop_floor_id, machine.work_center_idx, machine.machine_idx)] = TDSimulator.Record(
        #     state=result.state,
        #     action=result.action,
        #     next_state=None,
        #     reward=None,
        #     done=False
        # )

        return result.result

    def route(self, shop_floor_id: str, job: Job, work_center_idx: int, machines: List[Machine]) -> 'Machine | None':
        parameters = WorkCenterInput(job, work_center_idx, machines)
        result = self.work_center.schedule(parameters)

        return result.result

    def will_produce(self, shop_floor_id: str, job: Job, machine: Machine):
        pass

    def did_produce(self, shop_floor_id: str, job: Job, machine: Machine):
        pass
        # from reward.base.surrogate import Surrogate
        #
        # parameters = MachineInput(machine, self.environment.now)
        #
        # self.queue[
        #     TDSimulator.Key(shop_floor_id, machine.work_center_idx, machine.machine_idx)
        # ].next_state = self.machine.encode_state(parameters)
        #
        # if isinstance(self.machine_reward, Surrogate):
        #     self.queue[
        #         TDSimulator.Key(shop_floor_id, machine.work_center_idx, machine.machine_idx)
        #     ].reward = self.machine_reward.reward(job, machine)
        #
        # self.queue[
        #     TDSimulator.Key(shop_floor_id, machine.work_center_idx, machine.machine_idx)
        # ].next_state = self.machine.encode_state(parameters)
        #
        # if isinstance(self.machine_reward, Surrogate):
        #     self.queue[
        #         TDSimulator.Key(shop_floor_id, machine.work_center_idx, machine.machine_idx)
        #     ].reward = self.machine_reward.reward(job, machine)

    def will_dispatch(self, shop_floor_id: str, job: Job, work_center: WorkCenter):
        pass
        # self.queue[
        #     TDSimulator.Key(shop_floor_id, machine.work_center_idx, machine.machine_idx)
        # ].next_state = self.machine.encode_state(parameters)
        #
        # if isinstance(self.machine_reward, Surrogate):
        #     self.queue[
        #         TDSimulator.Key(shop_floor_id, machine.work_center_idx, machine.machine_idx)
        #     ].reward = self.machine_reward.reward(job, machine)

    def did_dispatch(self, shop_floor_id: str, job: Job, work_center: WorkCenter, machine: Machine):
        pass
        # self.queue[
        #     TDSimulator.Key(shop_floor_id, machine.work_center_idx, machine.machine_idx)
        # ].next_state = self.machine.encode_state(parameters)
        #
        # if isinstance(self.machine_reward, Surrogate):
        #     self.queue[
        #         TDSimulator.Key(shop_floor_id, machine.work_center_idx, machine.machine_idx)
        #     ].reward = self.machine_reward.reward(job, machine)


    def did_complete(self, shop_floor_id: str, job: Job):
        pass
        # from reward.base.global import Global
        #
        # if isinstance(self.machine_reward, Surrogate):
        #     self.queue[
        #         TDSimulator.Key(shop_floor_id, job.work_center_idx, -1)
        #     ].reward = self.work_center_reward.reward(job, self.environment.now)
        #
        # if isinstance(self.work_center_reward, Global):
        #     self.queue[
        #         TDSimulator.Key(shop_floor_id, job.work_center_idx, -1)
        #     ].reward = self.work_center_reward.reward(job, self.environment.now)

    def did_finish_simulation(self, shop_floor_id: str):
        pass
