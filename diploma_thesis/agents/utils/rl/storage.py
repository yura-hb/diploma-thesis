
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

    def sample(self, update_returns: bool, device: torch.device, batch_graphs: bool = True):
        batch, info = self.memory.sample(return_info=True)

        if self.is_episodic:
            return self.__process_episodic_batched_data__(batch, update_returns, batch_graphs, device), info

        return self.__process_batched_data__(batch, update_returns, batch_graphs, device), info

    def sample_minibatches(self, update_returns, device, n, sample_ratio):
        batch, info = self.sample(update_returns, device, batch_graphs=False)

        mask = torch.zeros(batch.batch_size, device=device)

        for i in range(n):
            mask_ = mask.uniform_() < sample_ratio
            idx = mask_.nonzero()

            sub_batch = batch[mask_]

            if isinstance(batch.state, GraphState):
                sub_batch.state.graph = Batch.from_data_list(
                    [batch.state.graph[index] for index in idx]
                ).to(device)

                sub_batch.next_state.graph = Batch.from_data_list(
                    [batch.next_state.graph[index] for index in idx]
                ).to(device)

            yield sub_batch

    def update_priority(self, indices: torch.LongTensor, priorities: torch.FloatTensor):
        self.memory.update_priority(indices, priorities)

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

    def __process_episodic_batched_data__(self, batch, update_returns: bool, batch_graphs, device: torch.device):
        batch = [element[0] for element in batch]

        if update_returns:
            batch = [self.return_estimator.update_returns(trajectory) for trajectory in batch]

        result = reduce(lambda x, y: x + y, batch, [])

        return self.__merge_batched_data__(result, batch_graphs, device)

    def __process_batched_data__(self, batch, update_returns: bool, batch_graphs, device: torch.device):
        batch = [element[0] for element in batch]

        if update_returns:
            batch = self.return_estimator.update_returns(batch)

        return self.__merge_batched_data__(batch, batch_graphs, device)

    def __merge_batched_data__(self, batch, batch_graphs, device):
        batch = [element.view(-1) for element in batch]

        if isinstance(batch[0].state, GraphState) and isinstance(batch[0].next_state, GraphState):
            state_graph = []
            next_state_graph = []

            for element in batch:
                state_graph += [element.state.graph]
                next_state_graph += [element.next_state.graph]

                element.state.graph, element.next_state.graph = None, None

            result = torch.cat(batch, dim=0)

            for index, element in enumerate(batch):
                element.state.graph, element.next_state.graph = state_graph[index], next_state_graph[index]

            result.state.graph = self.__collate_graphs__(state_graph, batch_graphs, device)
            result.next_state.graph = self.__collate_graphs__(next_state_graph, batch_graphs, device)
        else:
            result = torch.cat(batch, dim=0)

        return result.to(device)

    def __collate_graphs__(self, records: List[Graph], batch_graphs, device: torch.device):
        graphs = [record[0].to_pyg_graph() for record in records]

        if not batch_graphs:
            return graphs

        batch = Batch.from_data_list(graphs).to(device)

        return batch
