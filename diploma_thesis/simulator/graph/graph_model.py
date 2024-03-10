from dataclasses import dataclass
from typing import Dict

import torch
import torch_geometric as pyg

from agents.base import Graph
from environment import Job, Delegate, Context, WorkCenter, Machine
from utils import OrderedSet
from .transition import GraphTransition, from_cli
from .util import Encoder, EncoderConfiguration


@dataclass
class Configuration(EncoderConfiguration):
    memory: int = 10000

    @staticmethod
    def from_cli(parameters: Dict) -> 'Configuration':
        return Configuration(
            memory=parameters.get('memory', 0),
            is_machine_set_in_work_center_connected=parameters.get('is_machine_set_in_work_center_connected', False),
            is_work_center_set_in_shop_floor_connected=parameters.get(
                'is_work_center_set_in_shop_floor_connected', False
            )
        )


class GraphModel(Delegate):

    @dataclass
    class Record:
        graph: Graph
        previous_encoded_graph: Graph | None
        did_change_jobs: bool
        job_operation_map: Dict[torch.LongTensor, Dict[Graph.OperationKey, int]]
        completed_job_ids: OrderedSet
        n_ops: int

    def __init__(self, transition_model: GraphTransition, configuration: Configuration):
        super().__init__(children=[])

        self.configuration = configuration
        self.transition_model = transition_model
        self.encoder = Encoder(configuration)
        self.cache = {}
        self.job_operation_map: Dict[int, GraphModel.Record] = dict()

    def graph(self, context: Context) -> Graph | None:
        record = self.cache.get(context.shop_floor.id)

        if record is None:
            return None

        did_change_jobs = record.did_change_jobs
        record.did_change_jobs = False

        graph = record.graph
        previous_encoded_graph = record.previous_encoded_graph
        job_operation_map = record.job_operation_map

        new_encoded_graph = self.encoder.encode(
            previous=previous_encoded_graph,
            graph=graph,
            did_change_jobs=did_change_jobs,
            job_operation_map=job_operation_map,
            shop_floor=context.shop_floor
        )

        record.previous_encoded_graph = new_encoded_graph

        new_encoded_graph_data = new_encoded_graph.data

        self.__trim_empty_records__(new_encoded_graph_data)

        return Graph(data=new_encoded_graph_data.clone())

    def did_start_simulation(self, context: Context):
        self.cache[context.shop_floor.id] = GraphModel.Record(
            graph=Graph(),
            previous_encoded_graph=None,
            did_change_jobs=False,
            job_operation_map=dict(),
            completed_job_ids=OrderedSet(),
            n_ops=0
        )

    def will_dispatch(self, context: Context, job: Job, work_center: WorkCenter):
        sid = context.shop_floor.id

        if job.current_step_idx == 0 and sid in self.cache:
            self.cache[sid].graph = self.transition_model.append(job, context.shop_floor, self.cache[sid].graph)
            self.cache[sid].did_change_jobs = True
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

                self.cache[sid].graph = self.transition_model.remove(job, context.shop_floor, self.cache[sid])
                self.cache[sid].did_change_jobs = True
                self.__remove_job_from_operation_map__(context, job)

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
                    job_id=job.id.item(), work_center_id=work_center_id.item(), machine_id=machine_id
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

    def __trim_empty_records__(self, data: pyg.data.HeteroData):
        for key, store in data.node_items():
            if 'x' in store and store['x'].numel() == 0:
                del data[key]

                for edge, _ in data.edge_items():
                    if key in edge:
                        del data[edge]

    @staticmethod
    def from_cli(parameters: Dict) -> 'GraphModel':
        return GraphModel(transition_model=from_cli(parameters['transition_model']),
                          configuration=Configuration.from_cli(parameters))
