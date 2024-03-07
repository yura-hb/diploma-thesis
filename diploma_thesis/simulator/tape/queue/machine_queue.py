
import environment
import torch

from typing import Dict, List

from agents.machine.model import MachineModel
from environment import MachineKey
from simulator.tape.machine import MachineReward
from utils import filter
from .queue import *


class MachineQueue(Queue):

    def __init__(self, reward: MachineReward):
        super().__init__()

        self.reward = reward
        self.queue: Dict[MachineKey, List[TapeRecord]] = dict()

    # Preparation

    def prepare(self, shop_floor: ShopFloor):
        for machine in shop_floor.machines:
            self.queue[machine.key] = []

    # Reward Emit

    @filter(lambda self, _, __, record, ___: isinstance(record.result, Job))
    def register(self, context: Context, machine: Machine, record: MachineModel.Record, mode: NextStateRecordMode):
        if record.result is None:
            mode = NextStateRecordMode.on_next_action

        self.__record_next_state_on_action__(context, record.record.state, machine.key)
        self.__append_to_queue__(context, machine, record, mode)

    def did_produce(self, context: Context, machine: Machine, job: Job):
        record = self.queue[machine.key][-1]
        record.record.reward = self.reward.reward_after_production(record.context)

        if record.mode == NextStateRecordMode.on_produce:
            state = self.simulator.encode_machine_state(context=context, machine=machine)

            record.record.next_state = state

        self.__emit_rewards__(context, machine.work_center_idx, machine.machine_idx)

    def did_complete(self, context: Context, job: Job):
        records = self.__fetch_records_from_job_path__(context, job)

        if len(records) == 0:
            return

        rewards = self.reward.reward_after_completion([record.context for record in records])

        if rewards is None:
            return

        for index in rewards.indices:
            records[index].record.reward = rewards.reward[index]

        for unit in range(rewards.units.shape[1]):
            self.__emit_rewards__(context, rewards.units[0, unit], rewards.units[1, unit])

    # Utility

    def __record_next_state_on_action__(self, context, state, machine_key):
        if len(self.queue[machine_key]) == 0:
            return

        record = self.queue[machine_key][-1]

        if record.mode != NextStateRecordMode.on_next_action:
            return

        record.record.next_state = state

        self.__emit_rewards__(context, machine_key.work_center_id, machine_key.machine_id)

    def __append_to_queue__(
        self, context: Context, machine: Machine, record: MachineModel.Record, mode: NextStateRecordMode
    ):
        mid = machine.key

        self.queue[mid] += [TapeRecord(
            job_id=record.result.id if record.result is not None else None,
            record=Record(
                state=record.record.state,
                action=record.record.action,
                next_state=None,
                reward=None,
                done=torch.tensor(False, dtype=torch.bool),
                info=record.record.info,
                batch_size=[]
            ),
            context=self.reward.record_job_action(record.result, machine, context.moment),
            moment=context.moment,
            mode=mode
        )]

    def __fetch_records_from_job_path__(self, context: Context, job: Job):
        records = []

        def fn(index, machine):
            nonlocal records

            for record in self.queue[machine.key]:
                is_target_job = record.job_id == job.id

                if is_target_job:
                    records += [record]
                    continue

                is_idle_with_job_in_queue = (record.job_id is None and
                                             job.history.arrived_at_machine[index] <= record.moment <
                                             job.history.started_at[index])

                if is_idle_with_job_in_queue:
                    records += [record]
                    continue

        self.__enumerate_job_path__(context, job, fn)

        return records

    def __emit_rewards__(self, context: Context, work_center_idx, machine_idx):
        machine = context.shop_floor.machine(work_center_idx, machine_idx)

        self.__emit_rewards_to_machine__(context, machine)

    def __emit_rewards_to_machine__(self, context: Context, machine: Machine):
        records = self.queue[machine.key]

        remove_idx = []

        for index, record in enumerate(records):
            result = record.record

            if not result.is_filled:
                continue

            remove_idx += [index]

            self.simulator.did_prepare_machine_record(
                context=environment.Context(shop_floor=context.shop_floor, moment=record.moment),
                machine=machine,
                record=result
            )

        for index in reversed(remove_idx):
            del self.queue[machine.key][index]

    @staticmethod
    def __enumerate_job_path__(context: Context, job: Job, fn):
        for index, _ in enumerate(job.step_idx):
            work_center_idx = job.step_idx[index]
            machine_idx = job.history.arrived_machine_idx[index]
            machine = context.shop_floor.machine(work_center_idx, machine_idx)

            fn(index, machine)
