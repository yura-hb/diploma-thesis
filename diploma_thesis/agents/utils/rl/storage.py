
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

    def sample(self, device: torch.device, merge: bool = True):
        batch, info = self.memory.sample(return_info=True)

        if self.is_episodic:
            return self.__process_episodic_batched_data__(batch, merge, device), info

        return self.__process_batched_data__(batch, merge, device), info

    def sample_minibatches(self, device, n, sample_count):
        batch, info = self.sample(device, merge=False)

        def generator(batch):
            for i in range(n):
                idx = torch.randint(0, len(batch), (sample_count,))

                minibatch = [batch[index] for index in idx]
                minibatch = self.__merge_batched_data__(minibatch, merge=True, device=device)
                #
                # print(f'{minibatch.info[Record.ADVANTAGE_KEY]}')
                # print(f'{minibatch.info[Record.RETURN_KEY]}')

                yield minibatch

                del minibatch

        def load_batch():
            return self.__merge_batched_data__(batch, merge=True, device=device)

        return load_batch, generator(batch), info

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

    def __process_episodic_batched_data__(self, batch, merge, device: torch.device):
        batch = [element[0] for element in batch]

        result = reduce(lambda x, y: x + y, batch, [])

        return self.__merge_batched_data__(result, merge, device)

    def __process_batched_data__(self, batch, merge, device: torch.device):
        batch = [element[0] for element in batch]

        return self.__merge_batched_data__(batch, merge, device)

    def __merge_batched_data__(self, batch, merge, device):
        if not merge:
            return batch

        batch = [element.view(-1) for element in batch]
        items = self.__collate_variable_length_info_values__(batch)

        self.__fill_empty_memory__(batch)

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

            result.state.graph = self.__collate_graphs__(state_graph, merge, device)
            result.next_state.graph = self.__collate_graphs__(next_state_graph, merge, device)
        else:
            result = torch.cat(batch, dim=0)

        for key, value in items.items():
            result.info[key] = value

        return result.to(device)

    def __fill_empty_memory__(self, batch):
        non_zero_memory = [element for element in batch if element.state.memory is not None]

        # Some records can have no memory
        if len(non_zero_memory) > 0:
            non_zero_memory = torch.zeros_like(non_zero_memory[0].state.memory)

            for element in batch:
                if element.state.memory is None:
                    element.state.memory = non_zero_memory

                if element.next_state.memory is None:
                    element.next_state.memory = non_zero_memory

    def __collate_variable_length_info_values__(self, batch):
        keys = [Record.POLICY_KEY, Record.ACTION_KEY, Record.VALUE_KEY]
        fill_values = {Record.POLICY_KEY: 0.0,
                       Record.ACTION_KEY: torch.finfo(torch.float32).min,
                       Record.VALUE_KEY: torch.finfo(torch.float32).min}

        result = dict()

        for key in keys:
            result[key] = []

        for element in batch:
            for key in keys:
                result[key] += torch.atleast_2d(element.info[key])

        for key in keys:
            result[key] = torch.nn.utils.rnn.pad_sequence(result[key], batch_first=True, padding_value=fill_values[key])

        # In some cases batch can be sampled with repetition
        for element in batch:
            keys_ = element.info.keys()

            for key in keys:
                if key in keys_:
                    del element.info[key]

        return result

    def __collate_graphs__(self, records: List[Graph], merge, device: torch.device):
        graphs = [record.to_pyg_graph() for record in records]

        if not merge:
            return graphs

        batch = Batch.from_data_list(graphs).to(device)

        return batch
