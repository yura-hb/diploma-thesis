from typing import Dict, List

import torch

import environment
from agents.workcenter.model import WorkCenterModel
from environment import WorkCenter, WorkCenterKey
from simulator.tape.work_center import WorkCenterReward
from .queue import *


class WorkCenterQueue(Queue):

    def __init__(self, reward: WorkCenterReward):
        super().__init__()

        self.reward = reward
        self.queue: Dict[WorkCenterKey, List[TapeRecord]] = dict()

    # Preparation

    def prepare(self, shop_floor: ShopFloor):
        self.queue = dict()

        for work_center in shop_floor.work_centers:
            self.queue[work_center.key] = []

    # Utility

    def register(self,
                 context: Context,
                 work_center: WorkCenter,
                 job: Job,
                 record: WorkCenterModel.Record,
                 mode: NextStateRecordMode):
        if record.result is None:
            mode = NextStateRecordMode.on_next_action

        self.__record_next_state_on_action__(record.record.state, work_center.key)
        self.__append_to_queue__(context, work_center, job, record, mode)

    def did_produce(self, context: Context, machine: Machine, job: Job, is_naive_decision: bool):
        record = self.queue[machine.work_center.key][-1]
        record.record.reward = self.reward.reward_after_production(record.context)

        if record.record.reward is not None:
            self.__emit_rewards__(context, machine.work_center_idx)
            return

        if record.mode != NextStateRecordMode.on_produce:
            return

        state = self.simulator.encode_work_center_state(context=context,
                                                        work_center=machine.work_center,
                                                        job=job,
                                                        memory=record.memory)

        record.record.next_state = state

    def did_complete(self, context: Context, job: Job):
        records = self.__fetch_records_from_job_path__(context, job)

        if len(records) == 0:
            return

        rewards = self.reward.reward_after_completion([record.context for record in records])

        if rewards is None:
            return

        for index in rewards.indices:
            records[index].record.reward = rewards.reward[index]

        for unit in range(rewards.work_center_idx.shape[0]):
            self.__emit_rewards__(context, rewards.work_center_idx[unit])

    # Utils

    def __record_next_state_on_action__(self, state, key):
        if len(self.queue[key]) == 0:
            return

        record = self.queue[key][-1]

        if record.mode != NextStateRecordMode.on_next_action:
            return

        record.record.next_state = state

    def __append_to_queue__(
            self,
            context: Context,
            work_center: WorkCenter,
            job: Job,
            record: WorkCenterModel.Record,
            mode: NextStateRecordMode
    ):
        wid = work_center.key

        self.queue[wid] += [TapeRecord(
            job_id=job.id,
            record=Record(
                state=record.record.state,
                action=record.record.action,
                next_state=None,
                info=record.record.info,
                reward=None,
                done=torch.tensor(False, dtype=torch.bool),
            ),
            context=self.reward.record_job_action(record.result, work_center),
            memory=record.record.memory,
            moment=context.moment,
            mode=mode
        )]

    def __fetch_records_from_job_path__(self, context: Context, job: Job):
        records = []

        def fn(index, work_center):
            nonlocal records

            for record in self.queue[work_center.key]:
                is_target_job = record.job_id == job.id

                if is_target_job:
                    records += [record]
                    continue

                is_idle_with_job_in_queue = (record.job_id is None and
                                             job.history.arrived_at_work_center[index] <= record.moment <
                                             job.history.arrived_at_machine[index])

                if is_idle_with_job_in_queue:
                    records += [record]
                    continue

        self.__enumerate_job_path__(context, job, fn)

        return records

    def __emit_rewards__(self, context: Context, work_center_idx):
        work_center = context.shop_floor.work_center(work_center_idx)

        self.__emit_reward_to_work_center__(context, work_center)

    def __emit_reward_to_work_center__(self, context: Context, work_center: WorkCenter):
        records = self.queue[work_center.key]

        remove_idx = []

        for index, record in enumerate(records):
            result = record.record

            if not result.is_filled:
                continue

            remove_idx += [index]

            self.simulator.did_prepare_work_center_record(
                context=environment.Context(shop_floor=context.shop_floor, moment=record.moment),
                work_center=work_center,
                record=result
            )

        for index in reversed(remove_idx):
            del self.queue[work_center.key][index]

    @staticmethod
    def __enumerate_job_path__(context: Context, job: Job, fn):
        for index, _ in enumerate(job.step_idx):
            work_center_idx = job.step_idx[index]
            work_center = context.shop_floor.work_center(work_center_idx)

            fn(index, work_center)
