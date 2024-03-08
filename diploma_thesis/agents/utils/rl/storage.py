
import torch

from typing import List

from agents.base.state import GraphState, Graph
from agents.utils.memory import Memory, Record
from agents.base.agent import TrainingSample, Trajectory, Slice
from agents.utils.return_estimator import ReturnEstimator

from functools import reduce

from torch_geometric.data import Batch


class Storage:

    def __init__(self, is_episodic: bool, memory: Memory, return_estimator: ReturnEstimator):
        self.is_episodic = is_episodic
        self.memory = memory
        self.return_estimator = return_estimator

    def store(self, sample: TrainingSample):
        records = self.__prepare__(sample)

        for record in records:
            record.info['episode'] = sample.episode_id

        if self.is_episodic:
            self.memory.store([records])
        else:
            records = [record.view(-1) for record in records]

            self.memory.store(records)

    def sample(self, update_returns: bool = True):
        batch, info = self.memory.sample(return_info=True)

        if self.is_episodic:
            return self.__process_episodic_batched_data__(batch, update_returns), info

        return self.__process_batched_data__(batch, update_returns), info

    def clear(self):
        self.memory.clear()

    def __prepare__(self, sample: TrainingSample) -> List[Record]:
        match sample:
            case Trajectory(_, records):
                updated = self.return_estimator.update_returns(records)
                updated = [record for record in updated]

                return updated
            case Slice(_, records):
                if len(records) == 1:
                    return [records[0]]

                updated = self.return_estimator.update_returns(records)

                return [updated[0]]
            case _:
                raise ValueError(f'Unknown sample type: {type(sample)}')

    def __process_episodic_batched_data__(self, batch, update_returns: bool):
        if update_returns:
            batch = [
                [self.return_estimator.update_returns(trajectory) for trajectory in element] for element in batch
            ]

        batch = [element[0] for element in batch]

        result = [torch.cat([step.view(-1) for step in trajectory], dim=0) for trajectory in batch]
        result = torch.cat(result, dim=0)

        if isinstance(batch[0][0].state, GraphState) and isinstance(batch[0][0].next_state, GraphState):
            records = reduce(lambda x, y: x + y, batch)

            result.state.graph = Graph(self.__collate_graphs__([record.state.graph for record in records]))
            result.next_state.graph = Graph(self.__collate_graphs__([record.next_state.graph for record in records]))

        return result

    def __process_batched_data__(self, batch, update_returns: bool):
        if update_returns:
            batch = self.return_estimator.update_returns(batch)

        result = torch.cat([element.view(-1) for element in batch], dim=0)

        if isinstance(batch[0].state, GraphState) and isinstance(batch[0].next_state, GraphState):
            result.state.graph.data = self.__collate_graphs__([record.state.graph for record in batch])
            result.next_state.graph.data = self.__collate_graphs__([record.next_state.graph for record in batch])

        return result

    def __collate_graphs__(self, records: List[Graph]):
        data = reduce(lambda x, y: x + y, (record.data.to_data_list() for record in records))

        return Batch.from_data_list(data)
