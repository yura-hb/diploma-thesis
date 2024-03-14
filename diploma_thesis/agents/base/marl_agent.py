import copy
from typing import Dict

import pandas as pd

from .rl_agent import *


class MARLAgent(Generic[Key], RLAgent[Key]):

    def __init__(self, is_model_distributed: bool, *args, **kwargs):
        self.is_configured = False

        super().__init__(*args, **kwargs)

        self.model: DeepPolicyModel | Dict[Key, DeepPolicyModel] = self.model
        self.trainer: RLTrainer | Dict[Key, RLTrainer] = self.trainer
        self.is_model_distributed = is_model_distributed
        self.keys = None

    def setup(self, shop_floor: ShopFloor):
        if self.is_configured:
            is_key_set_equal = self.keys == list(self.iterate_keys(shop_floor))
            is_evaluating_with_centralized_model = self.phase == EvaluationPhase() and not self.is_model_distributed

            assert is_key_set_equal or is_evaluating_with_centralized_model, \
                ("Multi-Agent policy should be configured for the same shop floor architecture "
                 "or have centralized action network")

            return

        self.is_configured = True
        self.keys = list(self.iterate_keys(shop_floor))

        base_model = self.model
        base_trainer = self.trainer

        self.model = dict() if self.is_model_distributed else self.model
        self.trainer = dict()

        for key in self.keys:
            if self.is_model_distributed:
                self.model[key] = copy.deepcopy(base_model)
                self.model.configure(self.configuration)

            self.trainer[key] = copy.deepcopy(base_trainer)

        if not self.is_model_distributed:
            self.model.configure(self.configuration)

    def schedule(self, key: Key, parameters):
        state = self.encode_state(parameters)
        model = self.__model_for_key__(key)
        result = model(state, parameters)

        if not self.trainer[key].is_configured:
            self.trainer[key].configure(model.policy, configuration=self.configuration)

        return result

    @property
    def is_trainable(self):
        return True

    @property
    def is_distributed(self):
        return True

    @abstractmethod
    def iterate_keys(self, shop_floor: ShopFloor):
        pass

    @filter(lambda self: self.phase == TrainingPhase())
    def train_step(self):
        for key in self.keys:
            self.trainer[key].train_step(self.__model_for_key__(key).policy)

    @filter(lambda self, *args: self.phase == TrainingPhase())
    def store(self, key: Key, sample: TrainingSample):
        self.trainer[key].store(sample, self.__model_for_key__(key).policy)

    def loss_record(self):
        if self.keys is None:
            return pd.DataFrame()

        result = []

        for key in self.keys:
            loss_record = self.trainer[key].loss_record()

            for k, v in key.__dict__.items():
                loss_record[k] = int(v)

            result += [loss_record]

        return pd.concat(result)

    def clear_memory(self):
        if self.keys is None:
            return

        for key in self.keys:
            self.trainer[key].clear()

    def __model_for_key__(self, key: Key):
        if self.is_model_distributed:
            return self.model[key]

        return self.model

    def state_dict(self):
        state_dict = dict(is_configured=self.is_configured, keys=self.keys)

        if self.is_configured:
            model = {key: model.state_dict() for key, model in self.model.item()} \
                     if self.is_model_distributed else self.model.state_dict()

            state_dict.update(dict(
                model=model,
                trainer={key: trainer.state_dict() for key, trainer in self.trainer.items()}
            ))

            return state_dict

        state_dict.update(
            model=self.model.state_dict(),
            trainer=self.trainer.state_dict()
        )

        return state_dict

    def load_state_dict(self, state_dict: dict):
        self.is_configured = state_dict['is_configured']
        self.keys=self.keys

        if self.is_configured:
            for key in self.keys:
                if self.is_model_distributed:
                    self.model[key].load_state_dict(state_dict['model'][key])

                self.trainer[key].load_state_dict(state_dict['trainer'][key])

            if not self.is_model_distributed:
                self.model.load_state_dict(state_dict['model'])

            return

        self.model.load_state_dict(state_dict['model'])
        self.trainer.load_state_dict(state_dict['trainer'])
