from typing import Dict

from agents import MachineInput
from agents.machine.model import MachineModel
from agents.utils.memory import Record
from environment import MachineKey
from tape.machine import MachineReward
from .queue import *
from .tape_record import TapeRecord


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

    # Computation

    def register(self, shop_floor: ShopFloor, machine: Machine, record: MachineModel.Record):
        if isinstance(record.result, Job):
            self.queue[shop_floor.id][machine.key][record.result.id] = TapeRecord(
                record=Record(
                    state=record.state,
                    action=record.action
                ),
                context=self.reward.record_job_action(record.result, machine)
            )

    def record_state(self, context: DelegateContext, machine: Machine, job: Job):
        if job.id not in self.queue[context.shop_floor.id][machine.key]:
            return

        parameters = MachineInput(machine, context.moment)
        state = self.simulator.encode_machine_state(parameters)

        self.queue[context.shop_floor.id][machine.key][job.id].record.next_state = state

    def emit_intermediate_reward(self, context: DelegateContext, machine: Machine, job: Job):
        if job.id not in self.queue[context.shop_floor.id][machine.key]:
            return

        record = self.queue[context.shop_floor.id][machine.key][job.id]
        context = record.context
        reward = self.reward.reward_after_production(context)

        if reward is None:
            return

        record = record.record
        record.reward = reward

        self.simulator.did_prepare_machine_reward(context.shop_floor, machine, record)

        del self.queue[context.shop_floor.id][machine.key][job.id]

    def emit_reward_after_completion(self, context: DelegateContext, job: Job):
        contexts = self.__fetch_contexts_from_job_path__(context, job)

        if len(contexts) != len(job.step_idx):
            return

        reward = self.reward.reward_after_completion(contexts)

        if reward is None:
            return

    def __fetch_contexts_from_job_path__(self, context: DelegateContext, job: Job):
        contexts = []

        for index, step in enumerate(job.step_idx):
            work_center_idx = job.step_idx[index]
            machine_idx = job.history.arrived_machine_idx[index]
            machine = context.shop_floor.machine(work_center_idx, machine_idx)

            if context not in self.queue[context.shop_floor.id][machine.key]:
                continue

            contexts += [self.queue[context.shop_floor.id][machine.key][context]]

        return contexts
