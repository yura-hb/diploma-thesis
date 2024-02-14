from typing import Dict

from agents import MachineInput
from agents.machine.model import MachineModel
from agents.utils.memory import Record
from environment import MachineKey
from tape.machine import MachineReward
from tape.queue.queue import *
from tape.utils.tape_record import TapeRecord


class MachineQueue(Queue):

    def __init__(self, reward: MachineReward):
        super().__init__()
        self.reward = reward
        self.queue: Dict[ShopFloorId, Dict[MachineKey, Dict[ActionId, TapeRecord]]] = dict()

    # Preparation

    def prepare(self, shop_floor: ShopFloor):
        self.queue[shop_floor.id] = dict()

        for machine in shop_floor.machines:
            self.queue[shop_floor.id][machine.key] = dict()

    def clear(self, shop_floor: ShopFloor):
        del self.queue[shop_floor.id]

    def register(self, shop_floor: ShopFloor, machine: Machine, record: MachineModel.Record):
        if isinstance(record.result, Job):
            self.queue[shop_floor.id][machine.key][record.result.id] = TapeRecord(
                record=Record(
                    state=record.state,
                    action=record.action,
                    next_state=None,
                    reward=None,
                    done=False,
                    batch_size=[]
                ),
                context=self.reward.record_job_action(record.result, machine)
            )

    def record_next_state(self, context: DelegateContext, machine: Machine, job: Job):
        if job.id not in self.queue[context.shop_floor.id][machine.key]:
            return

        parameters = MachineInput(machine, context.moment)
        state = self.simulator.encode_machine_state(parameters)

        self.queue[context.shop_floor.id][machine.key][job.id].record.next_state = state

    def emit_intermediate_reward(self, context: DelegateContext, machine: Machine, job: Job):
        record = self.queue[context.shop_floor.id][machine.key].get(job.id)

        if record is None or record.context is None:
            return

        reward = self.reward.reward_after_production(record.context)

        if reward is None:
            return

        self.__emit_reward_to_machine__(context, machine, job, reward)

    def emit_reward_after_completion(self, context: DelegateContext, job: Job):
        contexts = self.__fetch_contexts_from_job_path__(context, job)

        if len(contexts) == 0:
            return

        rewards = self.reward.reward_after_completion(contexts)

        if rewards is None:
            return

        for reward in rewards:
            self.__emit_reward__(context, reward.work_center_idx, reward.machine_idx, job, reward.reward)

    # Utility

    def __fetch_contexts_from_job_path__(self, context: DelegateContext, job: Job):
        contexts = []

        def fn(index, machine):
            nonlocal contexts

            record = self.queue[context.shop_floor.id][machine.key].get(job.id)

            if record is None or record.context is None:
                return

            contexts += [record.context]

        self.__enumerate_job_path__(context, job, fn)

        return contexts

    def __emit_reward__(self, context: DelegateContext,
                        work_center_idx: int,
                        machine_idx: int,
                        job: Job,
                        reward: torch.FloatTensor):
        machine = context.shop_floor.machine(work_center_idx, machine_idx)

        self.__emit_reward_to_machine__(context, machine, job, reward)

    def __emit_reward_to_machine__(self,
                                   context: DelegateContext,
                                   machine: Machine,
                                   job: Job,
                                   reward: torch.FloatTensor):
        record = self.queue[context.shop_floor.id][machine.key].get(job.id)

        if record is None:
            return

        record = record.record
        record.reward = reward

        self.simulator.did_prepare_machine_record(context.shop_floor, machine, record)

        del self.queue[context.shop_floor.id][machine.key][job.id]

    @staticmethod
    def __enumerate_job_path__(context: DelegateContext, job: Job, fn):
        for index, _ in enumerate(job.step_idx):
            work_center_idx = job.step_idx[index]
            machine_idx = job.history.arrived_machine_idx[index]
            machine = context.shop_floor.machine(work_center_idx, machine_idx)

            fn(index, machine)
