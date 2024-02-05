
import torch

from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic, Any

Entity = TypeVar('Entity')
State = TypeVar('State')


class StateEncoder(Generic[Entity], metaclass=ABCMeta):

    @abstractmethod
    @property
    def is_trainable(self) -> bool:
        pass

    @abstractmethod
    def encode(self, entity: Entity) -> State:
        pass
