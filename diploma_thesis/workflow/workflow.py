import logging
import sys
import torch
from abc import ABCMeta, abstractmethod

import simpy


class Workflow(metaclass=ABCMeta):

    @abstractmethod
    def run(self):
        pass

    def __make_logger__(self, name: str, environment: simpy.Environment, log_stdout: bool) -> logging.Logger:
        class _Formatter(logging.Formatter):

            def format(self, record):
                time = environment.now

                if isinstance(time, torch.Tensor):
                    time = int(time.item())

                record.time = str(time)
                return super(_Formatter, self).format(record)

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        if log_stdout:
            formatter = _Formatter('%(asctime)s | %(time)s | %(name)s | %(levelname)s | %(message)s')
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setLevel(logging.DEBUG)
            stdout_handler.setFormatter(formatter)

            logger.addHandler(stdout_handler)

        return logger
