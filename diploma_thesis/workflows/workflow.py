
from abc import ABCMeta, abstractmethod


class Workflow(metaclass=ABCMeta):

    @abstractmethod
    def run(self):
        pass