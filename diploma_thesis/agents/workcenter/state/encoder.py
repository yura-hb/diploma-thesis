
from abc import ABCMeta

import torch
import torch_geometric as pyg

from agents.base import Encoder, State, GraphEncoder, Graph
from agents.workcenter.utils import Input as WorkCenterInput


class StateEncoder(Encoder[WorkCenterInput], metaclass=ABCMeta):

    Input = WorkCenterInput
    State = State


class GraphStateEncoder(GraphEncoder, metaclass=ABCMeta):

    def __localize__(self, parameters: StateEncoder.Input, graph: Graph):
        job_ids = torch.cat([job.id.view(-1) for job in parameters.work_center.state.queue])
        job_ids = torch.cat([job_ids, parameters.work_center.history.processed_job_ids])

        return super().__localize_with_job_ids__(graph, job_ids)

    def __post_encode__(self, graph: pyg.data.HeteroData, parameters: StateEncoder.Input) -> pyg.data.HeteroData:
        is_target = graph[Graph.JOB_INDEX_MAP][:, [2]] == parameters.work_center.work_center_idx

        graph[Graph.OPERATION_KEY][Graph.TARGET_KEY] = is_target.view(-1)

        return graph
