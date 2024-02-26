import environment

from environment import Job, Context


class Delegate:

    def __init__(self, children=None):
        if children is None:
            children = []

        self.children = children

    def did_start_simulation(self, context: Context):
        """
        Will be triggered after the start of run
        """
        for child in self.children:
            child.did_start_simulation(context=context)

    def will_produce(self, context: Context, job: Job, machine: 'environment.Machine'):
        """
        Will be triggered before the production of job on machine
        """
        for child in self.children:
            child.will_produce(context=context, job=job, machine=machine)

    def did_produce(self, context: Context, job: Job, machine: 'environment.Machine'):
        """
        Will be triggered after the production of job on machine
        """
        for child in self.children:
            child.did_produce(context=context, job=job, machine=machine)

    def will_dispatch(self, context: Context, job: Job, work_center: 'environment.WorkCenter'):
        """
        Will be triggered before dispatch of job on the work-center
        """
        for child in self.children:
            child.will_dispatch(context=context, job=job, work_center=work_center)

    def did_dispatch(self,
                     context: Context,
                     job: Job,
                     work_center: 'environment.WorkCenter',
                     machine: 'environment.Machine'):
        """
        Will be triggered after the dispatch of job to the machine
        """
        for child in self.children:
            child.did_dispatch(context=context, job=job, work_center=work_center, machine=machine)

    def did_finish_dispatch(self, context: Context, work_center: 'environment.WorkCenter'):
        """
        Will be triggered after the dispatch of job on the work-center
        """
        for child in self.children:
            child.did_finish_dispatch(context=context, work_center=work_center)

    def did_complete(self, context: Context, job: Job):
        """
        Will be triggered after the completion of job
        """
        for child in self.children:
            child.did_complete(context=context, job=job)

    def did_breakdown(self, context: Context, machine: 'environment.Machine', repair_time: float):
        """
        Will be triggered after the breakdown of machine
        """
        for child in self.children:
            child.did_breakdown(context=context, machine=machine, repair_time=repair_time)

    def did_repair(self, context: Context, machine: 'environment.Machine'):
        """
        Will be triggered after the repair of machine
        """
        for child in self.children:
            child.did_repair(context=context, machine=machine)

    def did_finish_simulation(self, context: Context):
        """
        Will be triggered after all jobs have been completed
        """
        for child in self.children:
            child.did_finish_simulation(context=context)
