from typing import Dict

from agents import WorkCenterInput
from agents.utils.memory import Record
from agents.workcenter.model import WorkCenterModel
from environment import WorkCenter, WorkCenterKey
from tape.work_center import WorkCenterReward
from .queue import *
from .tape_record import TapeRecord


class WorkCenterQueue(Queue):

    def __init__(self, reward: WorkCenterReward):
        super().__init__()

        self.reward = reward
        self.queue: Dict[ShopFloorId, Dict[WorkCenterKey, Dict[ActionId, TapeRecord]]] = dict()

    # Preparation

    def prepare(self, shop_floor: ShopFloor):
        self.queue[shop_floor.id] = dict()

        for machine in shop_floor.machines:
            self.queue[shop_floor.id][machine.key] = dict()

    def clear(self, shop_floor: ShopFloor):
        del self.queue[shop_floor.id]

    # Utility

    def register(self, shop_floor: ShopFloor, job: Job, work_center: WorkCenter, record: WorkCenterModel.Record):
        if isinstance(record.result, Job):
            self.queue[shop_floor.id][work_center.key][record.result.id] = TapeRecord(
                record=Record(
                    state=record.state,
                    action=record.action
                ),
                context=self.reward.record_job_action(record.result, work_center)
            )

    def record_state(self, context: DelegateContext, machine: Machine, job: Job):
        work_center = machine.work_center

        if job.id not in self.queue[context.shop_floor.id][work_center.key]:
            return

        parameters = WorkCenterInput(job, machine.work_center_idx, machine.work_center.machines)
        state = self.simulator.encode_work_center_state(parameters)

        self.queue[context.shop_floor.id][work_center.key][job.id].record.next_state = state

    def emit_intermediate_reward(self, context: DelegateContext, machine: Machine, job: Job):
        work_center = machine.work_center

        if job.id not in self.queue[context.shop_floor.id][work_center.key]:
            return

        record = self.queue[context.shop_floor.id][work_center.key][job.id]
        context = record.queue
        reward = self.reward.reward_after_production(context)

        if reward is not None:
            return

        record = record.record
        record.reward = reward

        self.simulator.did_prepare_work_center_reward(context.shop_floor, work_center, record)

        del self.queue[context.shop_floor.id][work_center.key][job.id]

    def emit_reward_after_completion(self, context: DelegateContext, job: Job):
        contexts = self.__fetch_contexts_from_job_path__(context, job)

        if len(contexts) != len(job.step_idx):
            return

        reward = self.reward.reward_after_completion(contexts)

    def __fetch_contexts_from_job_path__(self, context: DelegateContext, job: Job):
        contexts = []

        for index, step in enumerate(job.step_idx):
            work_center_idx = job.step_idx[index]
            work_center = context.shop_floor.work_center(work_center_idx)

            if context not in self.queue[context.shop_floor.id][work_center.key]:
                continue

            contexts += [self.queue[context.shop_floor.id][work_center.key][context]]

        return contexts
