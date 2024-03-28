import logging
import os
import shutil
import sys
from abc import ABCMeta, abstractmethod

import simpy
import torch


class Workflow(metaclass=ABCMeta):

    @abstractmethod
    def run(self):
        pass

    @property
    @abstractmethod
    def workflow_id(self) -> str:
        pass

    def __make_logger__(self, name: str, filename: str = None, log_stdout: bool = False) -> logging.Logger:
        logger = self.__get_logger__(name)

        formatter = logging.Formatter('%(asctime)s | | %(name)s | %(levelname)s | %(message)s')

        self.__add_handlers__(logger, formatter, filename, log_stdout)

        return logger

    def __make_time_logger__(self,
                             name: str,
                             environment: simpy.Environment,
                             filename: str = None,
                             log_stdout: bool = False) -> logging.Logger:
        class _Formatter(logging.Formatter):

            def format(self, record):
                time = environment.now

                if torch.is_tensor(time):
                    time = int(time.item())

                record.time = str(time)
                return super(_Formatter, self).format(record)

        logger = self.__get_logger__(name)

        formatter = _Formatter('%(asctime)s | %(time)s | %(name)s | %(levelname)s | %(message)s')

        self.__add_handlers__(logger, formatter, filename, log_stdout)

        return logger

    def __add_handlers__(self, logger, formatter, filename: str, log_stdout: bool):
        if log_stdout:
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setLevel(logging.DEBUG)
            stdout_handler.setFormatter(formatter)

            logger.addHandler(stdout_handler)

        if filename:
            dirname = os.path.dirname(filename)

            if not os.path.exists(dirname):
                os.makedirs(dirname)

            file_handler = logging.FileHandler(filename)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)

            logger.addHandler(file_handler)

    def __get_logger__(self, name):
        workflow_id = self.workflow_id

        logger = logging.getLogger(name + '_' + self.workflow_id if len(workflow_id) > 0 else name)
        logger.setLevel(logging.INFO)

        return logger

    def __make_output_dir__(self, name: str, output_dir: str, remove: bool = True):
        output_path = os.path.join(output_dir, name)
        exists = os.path.exists(output_path)

        if not exists:
            os.makedirs(output_path)
            return output_path

        if remove:
            shutil.rmtree(output_path)

        return output_path

    @staticmethod
    def __get_n_threads__(n_workers, n_threads):
        import multiprocessing as mp

        cpus = n_threads or mp.cpu_count()

        # Limit the number of threads to be at max of the number of cpu cores. Otherwise, torch oversubscription
        # cause significant slowdown https://github.com/pytorch/pytorch/issues/44025
        threads = max(1, cpus // n_workers if n_workers > 0 else cpus)

        return threads
