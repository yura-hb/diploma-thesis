import environment

from abc import ABCMeta
from environment import Job, Context


class Delegate(metaclass=ABCMeta):

    def did_start_simulation(self, context: Context):
        """
        Will be triggered after the start of run
        """
        pass

    def will_produce(self, context: Context, job: Job, machine: 'environment.Machine'):
        """
        Will be triggered before the production of job on machine
        """
        pass

    def did_produce(self, context: Context, job: Job, machine: 'environment.Machine'):
        """
        Will be triggered after the production of job on machine
        """
        pass

    def will_dispatch(self, context: Context, job: Job, work_center: 'environment.WorkCenter'):
        """
        Will be triggered before dispatch of job on the work-center
        """
        pass

    def did_dispatch(self,
                     context: Context,
                     job: Job,
                     work_center: 'environment.WorkCenter',
                     machine: 'environment.Machine'):
        """
        Will be triggered after the dispatch of job to the machine
        """
        pass

    def did_finish_dispatch(self, context: Context, work_center: 'environment.WorkCenter'):
        """
        Will be triggered after the dispatch of job on the work-center
        """
        pass

    def did_complete(self, context: Context, job: Job):
        """
        Will be triggered after the completion of job
        """
        pass

    def did_breakdown(self, context: Context, machine: 'environment.Machine', repair_time: float):
        """
        Will be triggered after the breakdown of machine
        """
        pass

    def did_repair(self, context: Context, machine: 'environment.Machine'):
        """
        Will be triggered after the repair of machine
        """
        pass

    def did_finish_simulation(self, context: Context):
        """
        Will be triggered after all jobs have been completed
        """
        pass
