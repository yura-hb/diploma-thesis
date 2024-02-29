from dataclasses import dataclass
from typing import Dict

import torch

import time

from agents.base import Graph
from environment import Job, MachineKey, WorkCenterKey, Delegate, Context, WorkCenter, Machine
from utils import OrderedSet
from .transition import GraphTransition, from_cli

from .util import Encoder


class GraphModel(Delegate):

    @dataclass
    class Record:
        graph: Graph
        job_operation_map: Dict[torch.LongTensor, Dict[Graph.OperationKey, int]]
        completed_job_ids: OrderedSet
        n_ops: int

    @dataclass
    class Configuration:
        memory: int = 10000

        @staticmethod
        def from_cli(parameters: Dict) -> 'GraphModel.Configuration':
            return GraphModel.Configuration(memory=parameters['memory'])

    def __init__(self, transition_model: GraphTransition, configuration: Configuration):
        super().__init__(children=[])

        self.configuration = configuration
        self.transition_model = transition_model
        self.cache = {}
        self.job_operation_map: Dict[int, GraphModel.Record] = dict()

    def graph(self, context: Context) -> Graph | None:
        record = self.cache.get(context.shop_floor.id)

        if record is None:
            return None

        return None

        # graph = record.graph
        # job_operation_map = record.job_operation_map
        #
        # encoder = Encoder()
        #
        # return encoder.encode(graph, context.shop_floor, job_operation_map, self.configuration)

    def did_start_simulation(self, context: Context):
        self.cache[context.shop_floor.id] = GraphModel.Record(
            graph=Graph(), job_operation_map=dict(), completed_job_ids=OrderedSet(), n_ops=0
        )

    def will_dispatch(self, context: Context, job: Job, work_center: WorkCenter):
        sid = context.shop_floor.id

        if job.current_step_idx == 0 and sid in self.cache:
            self.cache[sid].graph = self.transition_model.append(job, context.shop_floor, self.cache[sid].graph)
            self.__append_to_job_operation_map__(context, job)

    def did_dispatch(self, context: Context, job: Job, work_center: WorkCenter, machine: Machine):
        sid = context.shop_floor.id

        if sid in self.cache:
            self.cache[sid].graph = self.transition_model.update_on_dispatch(
                job, context.shop_floor, self.cache[sid].graph
            )

    def will_produce(self, context: Context, job: Job, machine: Machine):
        sid = context.shop_floor.id

        if sid in self.cache:
            self.cache[sid].graph = self.transition_model.update_on_will_produce(
                job, context.shop_floor, self.cache[sid].graph
            )

    def did_complete(self, context: Context, job: Job):
        sid = context.shop_floor.id

        if sid in self.cache:
            self.cache[sid].completed_job_ids.add(job.id)

            if len(self.cache[sid].completed_job_ids) > self.configuration.memory:
                job_id = self.cache[sid].completed_job_ids.pop(0)
                job = context.shop_floor.job(job_id)

                start = time.time()

                self.cache[sid] = self.transition_model.remove(job, context.shop_floor, self.cache[sid])
                self.__remove_job_from_operation_map__(context, job)

                print(f'GraphModel: Remove {time.time() - start}')

    def did_finish_simulation(self, context: Context):
        if context.shop_floor.id in self.cache:
            self.cache.pop(context.shop_floor.id)

    # Util

    def __append_to_job_operation_map__(self, context: Context, job: Job):
        sid = context.shop_floor.id

        self.cache[sid].job_operation_map[job.id] = dict()

        for step_id, work_center_id in enumerate(job.step_idx):
            for machine_id in range(len(job.processing_times[step_id])):
                operation_key = Graph.OperationKey(
                    job_id=torch.tensor(job.id, dtype=torch.long),
                    work_center_id=torch.tensor(work_center_id, dtype=torch.long),
                    machine_id=torch.tensor(machine_id, dtype=torch.long)
                )

                self.cache[sid].job_operation_map[job.id][operation_key] = self.cache[sid].n_ops
                self.cache[sid].n_ops += 1

    def __remove_job_from_operation_map__(self, context: Context, job: Job):
        sid = context.shop_floor.id

        n_removed_ops = len(self.cache[sid].job_operation_map.pop(job.id))
        self.cache[sid].n_ops -= n_removed_ops
        self.cache[sid].completed_job_ids.discard(job.id)

        keys = list(self.cache[sid].job_operation_map)
        keys = sorted(keys)

        for key in keys:
            # All jobs arrive at shop floor in ordered manner
            if key < job.id:
                continue

            self.cache[sid].job_operation_map[key] = {
                k: v - n_removed_ops for k, v in self.cache[sid].job_operation_map[key].items()
            }

            values = list(self.cache[sid].job_operation_map[key].values())
            max_key = max(values)
            min_key = min(values)

            assert max_key < self.cache[sid].n_ops and min_key >= 0, f'{max_key} {min_key} {self.cache[sid].n_ops}'


    @staticmethod
    def from_cli(parameters: Dict) -> 'GraphModel':
        return GraphModel(transition_model=from_cli(parameters['transition_model']),
                          configuration=GraphModel.Configuration.from_cli(parameters))
