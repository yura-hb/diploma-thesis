
from abc import abstractmethod
from enum import StrEnum
from typing import List

import pandas as pd
import torch

from agents.base.agent import TrainingSample, Trajectory, Slice
from agents.base.state import GraphState
from agents.utils.memory import Memory, Record
from agents.utils.nn import Loss, Optimizer
from agents.utils.policy import Policy
from agents.utils.return_estimator import ReturnEstimator
from utils import Loggable

from torch_geometric.data import Batch


class TrainSchedule(StrEnum):
    ON_TIMELINE = 'on_timeline'
    ON_STORE = 'on_store'
    ON_STORED_DATA_EXCLUSIVELY = 'on_stored_data_exclusively'

    @property
    def is_on_store(self):
        return self in [TrainSchedule.ON_STORE, TrainSchedule.ON_STORED_DATA_EXCLUSIVELY]

    @staticmethod
    def from_cli(parameters: dict):
        if 'train_schedule' not in parameters:
            return TrainSchedule.ON_TIMELINE

        return TrainSchedule(parameters['train_schedule'])


class RLTrainer(Loggable):

    def __init__(self,
                 memory: Memory,
                 loss: Loss,
                 optimizer: Optimizer,
                 return_estimator: ReturnEstimator,
                 train_schedule: TrainSchedule = TrainSchedule.ON_TIMELINE):
        super().__init__()

        self.memory = memory
        self.loss = loss
        self.optimizer = optimizer
        self.return_estimator = return_estimator
        self._is_configured = False
        self.train_schedule = train_schedule
        self.loss_cache = []

    @abstractmethod
    def configure(self, model: Policy):
        self._is_configured = True

        if not self.optimizer.is_connected:
            self.optimizer.connect(model.parameters())

    def __train__(self, model: Policy):
        pass

    @property
    def is_configured(self):
        return self._is_configured

    def train_step(self, model: Policy):
        if self.train_schedule != TrainSchedule.ON_TIMELINE:
            return

        self.__train__(model)

    def loss_record(self) -> pd.DataFrame:
        return pd.DataFrame(self.loss_cache)

    def store(self, sample: TrainingSample, model: Policy):
        records = self.__prepare__(sample)

        for record in records:
            record.info['episode'] = sample.episode_id

        # TODO: Store whole trajectory in memory

        records = [record.view(-1) for record in records]

        if self.train_schedule == TrainSchedule.ON_STORED_DATA_EXCLUSIVELY:
            self.memory.clear()

        self.memory.store(records)

        if not self.train_schedule.is_on_store:
            return

        self.__train__(model)

    def clear(self):
        self.loss_cache = []
        self.memory.clear()

    @staticmethod
    def from_cli(parameters, memory: Memory, loss: Loss, optimizer: Optimizer, return_estimator: ReturnEstimator):
        pass

    def record_loss(self, loss: torch.Tensor, **kwargs):
        self.loss_cache += [dict(
            value=loss.detach().item(),
            optimizer_step=self.optimizer.step_count,
            lr=self.optimizer.learning_rate,
            **kwargs
        )]

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

    def __sample_batch__(self, update_returns: bool = True):
        batch, info = self.memory.sample(return_info=True)

        if update_returns:
            batch = self.return_estimator.update_returns(batch)

        result = torch.cat(batch, dim=0)

        # Batch is the list of records. The main problematic use case is when batch contains records with
        # graph data, which we must merge into pytorch_geometric Batch object

        if isinstance(batch[0].state, GraphState) and isinstance(batch[0].next_state, GraphState):
            graphs = [record.state.graph.data for record in batch]
            next_state_graphs = [record.next_state.graph.data for record in batch]

            result.state.graph.data = Batch.from_data_list(graphs)
            result.next_state.graph.data = Batch.from_data_list(next_state_graphs)


        return batch, info
