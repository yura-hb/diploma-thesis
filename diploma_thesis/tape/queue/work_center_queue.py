from typing import Dict

from agents import WorkCenterInput
from agents.utils.memory import Record
from agents.workcenter.model import WorkCenterModel
from environment import WorkCenter, WorkCenterKey
from tape.work_center import WorkCenterReward
from tape.queue.queue import *
from tape.utils.tape_record import TapeRecord
from utils import filter


class WorkCenterQueue(Queue):

    def __init__(self, reward: WorkCenterReward):
        super().__init__()

        self.reward = reward
        self.queue: Dict[ShopFloorId, Dict[WorkCenterKey, Dict[ActionId, TapeRecord]]] = dict()

    # Preparation

    def prepare(self, shop_floor: ShopFloor):
        self.queue[shop_floor.id] = dict()

        for work_center in shop_floor.work_centers:
            self.queue[shop_floor.id][work_center.key] = dict()

    def clear(self, shop_floor: ShopFloor):
        del self.queue[shop_floor.id]

    def clear_all(self):
        self.queue = dict()

    # Utility

    @filter(lambda self, context, *args, **kwargs: context.shop_floor.id in self.queue)
    def register(self, context: Context, work_center: WorkCenter, job: Job,  record: WorkCenterModel.Record):
        self.queue[context.shop_floor.id][work_center.key][job.id] = TapeRecord(
            record=Record(
                state=record.state,
                action=record.action,
                next_state=None,
                reward=None,
                done=False,
            ),
            context=self.reward.record_job_action(record.result, work_center),
            moment=context.moment
        )

    @filter(lambda self, context, machine, job: job.id in self.queue[context.shop_floor.id][machine.work_center.key])
    def record_next_state(self, context: Context, machine: Machine, job: Job):
        work_center = machine.work_center
        parameters = WorkCenterInput(work_center=work_center, job=job)
        state = self.simulator.encode_work_center_state(parameters)

        self.queue[context.shop_floor.id][work_center.key][job.id].record.next_state = state

    @filter(lambda self, context, machine, job: job.id in self.queue[context.shop_floor.id][machine.work_center.key])
    def emit_intermediate_reward(self, context: Context, machine: Machine, job: Job):
        work_center = machine.work_center
        record = self.queue[context.shop_floor.id][work_center.key].get(job.id)

        reward = self.reward.reward_after_production(record.context)

        if reward is not None:
            return

        self.__emit_reward_to_work_center__(context, work_center, job, reward)

    def emit_reward_after_completion(self, context: Context, job: Job):
        contexts = self.__fetch_contexts_from_job_path__(context, job)

        if len(contexts) == 0:
            return

        reward = self.reward.reward_after_completion(contexts)

        if reward is None:
            return

        for record in reward:
            self.__emit_reward__(context, record.work_center_idx, job, record.reward)

    def __fetch_contexts_from_job_path__(self, context: Context, job: Job):
        contexts = []

        def fn(_, work_center):
            nonlocal contexts

            record = self.queue[context.shop_floor.id][work_center.key].get(job.id)

            if record is None or record.context is None:
                return

            contexts += [record.context]

        self.__enumerate_job_path__(context, job, fn)

        return contexts

    def __emit_reward__(self, context: Context,
                        work_center_idx: int,
                        job: Job,
                        reward: torch.FloatTensor):
        work_center = context.shop_floor.work_center(work_center_idx)

        self.__emit_reward_to_work_center__(context, work_center, job, reward)

    @filter(lambda self, context, work_center, job, _: job.id in self.queue[context.shop_floor.id][work_center.key])
    def __emit_reward_to_work_center__(self,
                                       context: Context,
                                       work_center: WorkCenter,
                                       job: Job,
                                       reward: torch.FloatTensor):
        record = self.queue[context.shop_floor.id][work_center.key].get(job.id)

        if record is None:
            return

        result = record.record
        result.reward = reward

        self.simulator.did_prepare_work_center_record(
            shop_floor=context.shop_floor,
            work_center=work_center,
            record=record,
            moment=record.moment
        )

        del self.queue[context.shop_floor.id][work_center.key][job.id]

    @staticmethod
    def __enumerate_job_path__(context: Context, job: Job, fn):
        for index, _ in enumerate(job.step_idx):
            work_center_idx = job.step_idx[index]
            work_center = context.shop_floor.work_center(work_center_idx)

            fn(index, work_center)
