
import torch

from typing import List, Tuple, Generator, Callable

from agents.base.state import Graph
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

            # Remove extra fields for training record
            if record.state.graph is not None and Graph.JOB_INDEX_MAP in record.state.graph.data.keys():
                del record.state.graph.data[Graph.JOB_INDEX_MAP]
                del record.next_state.graph.data[Graph.JOB_INDEX_MAP]

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

    def sample_minibatches(self, update_returns, device, n, sample_ratio) -> Tuple[Callable, Generator]:
        batch, info = self.sample(update_returns, device, batch_graphs=False)

        def generator(batch):
            mask = torch.zeros(batch.batch_size, device=device)

            for i in range(n):
                mask_ = mask.uniform_() < sample_ratio
                idx = mask_.nonzero()

                minibatch = batch[mask_]

                if batch[0].state.graph is not None:
                    minibatch.state.graph = Batch.from_data_list([
                        batch.state.graph[index] for index in idx
                    ]).to(device)

                    minibatch.next_state.graph = Batch.from_data_list(
                        [batch.next_state.graph[index] for index in idx]
                    ).to(device)

                yield minibatch

        def load_batch():
            if batch[0].state.graph is not None:
                batch.state.graph = Batch.from_data_list(batch.state.graph).to(device)
                batch.next_state.graph = Batch.from_data_list(batch.next_state.graph).to(device)

            return batch

        return load_batch, generator(batch)

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
        items = self.__collate_variable_length_info_values__(batch)

        if batch[0].state.memory is None and batch[1].state.memory is not None:
            batch[0].state.memory = torch.zeros_like(batch[1].state.memory)

        if batch[0].state.graph is not None:
            state_graph = []
            next_state_graph = []

            # Elements can be sampled with repetition
            for element in batch:
                state_graph += [element.state.graph[0]]
                next_state_graph += [element.next_state.graph[0]]

            for element in batch:
                element.state.graph, element.next_state.graph = None, None

            result = torch.cat(batch, dim=0)

            result.state.graph = self.__collate_graphs__(state_graph, batch_graphs, device)
            result.next_state.graph = self.__collate_graphs__(next_state_graph, batch_graphs, device)
        else:
            result = torch.cat(batch, dim=0)

        for key, value in items.items():
            result.info[key] = value

        return result.to(device)

    def __collate_variable_length_info_values__(self, batch):
        keys = [Record.POLICY_KEY, Record.ACTION_KEY]

        result = dict()

        for key in keys:
            result[key] = []

        for element in batch:
            for key in keys:
                result[key] += torch.atleast_2d(element.info[key])

        for key in keys:
            match key:
                case Record.POLICY_KEY:
                    fill_value = 0.0
                case Record.ACTION_KEY:
                    fill_value = torch.finfo(result[key][0].dtype).min

            result[key] = torch.nn.utils.rnn.pad_sequence(result[key], batch_first=True, padding_value=fill_value)

        # In some cases batch can be sampled with repetition
        for element in batch:
            for key in keys:
                if key in element.info.keys():
                    del element.info[key]

        return result

    def __collate_graphs__(self, records: List[Graph], batch_graphs, device: torch.device):
        graphs = [record.to_pyg_graph() for record in records]

        if not batch_graphs:
            return graphs

        batch = Batch.from_data_list(graphs).to(device)

        return batch
