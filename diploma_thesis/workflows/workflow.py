import logging
import sys
from abc import ABCMeta, abstractmethod


class Workflow(metaclass=ABCMeta):

    @abstractmethod
    def run(self):
        pass

    # TODO: Log environment time

    def __make_logger__(self, name: str = 'Workflow', log_stdout: bool = False) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')

        if log_stdout:
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setLevel(logging.DEBUG)
            stdout_handler.setFormatter(formatter)

            logger.addHandler(stdout_handler)

        return logger
