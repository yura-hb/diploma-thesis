import simpy
import torch

from environment import ShopFloor, Machine, Configuration
from .job_sampler import JobSampler
from .breakdown import Breakdown
from .initial_job_assignment import InitialJobAssignment

from .breakdown import from_cli as breakdown_from_cli
from .job_sampler import from_cli as job_sampler_from_cli
from .initial_job_assignment import from_cli as initial_job_assignment_from_cli


class ShopFloorManager:

    def __init__(self,
                 environment: simpy.Environment,
                 job_sampler: JobSampler,
                 breakdown: Breakdown,
                 initial_job_assignment: InitialJobAssignment,
                 seed: int = 0):
        self.environment = environment
        self.job_sampler = job_sampler
        self.breakdown = breakdown
        self.initial_job_assignment = initial_job_assignment
        self.seed = seed

    def simulate(self, shop_floor: ShopFloor):
        generator = torch.Generator()
        generator = generator.manual_seed(self.seed)

        self.breakdown.connect(generator)
        self.job_sampler.connect(generator)

        for job in self.initial_job_assignment.make_jobs(shop_floor, self.job_sampler):
            shop_floor.dispatch(job)

        self.environment.process(self.__dispatch__(shop_floor))

        for machine in shop_floor.machines:
            self.environment.process(self.__breakdown__(machine))

        return shop_floor.start()

    def __dispatch__(self, shop_floor: ShopFloor):
        while self.__should_dispatch__(shop_floor):
            arrival_time = self.job_sampler.sample_next_arrival_time()

            yield self.environment.timeout(arrival_time)

            job = self.job_sampler.sample(
                job_id=shop_floor.new_job_id,
                initial_work_center_idx=None,
                moment=self.environment.now)

            shop_floor.dispatch(job)

        shop_floor.did_finish_job_dispatch()

    def __breakdown__(self, machine: Machine):
        while True:
            if machine.state.will_breakdown:
                duration = max(machine.state.free_at - self.environment.now, 10)
                yield self.environment.timeout(duration)

            breakdown_arrival = self.breakdown.sample_next_breakdown_time(machine)

            yield self.environment.timeout(breakdown_arrival)

            repair_duration = self.breakdown.sample_repair_duration(self)

            machine.receive_breakdown_event(repair_duration)

    def __should_dispatch__(self, shop_floor: ShopFloor):
        number_of_jobs = self.job_sampler.number_of_jobs()

        if number_of_jobs is not None:
            return shop_floor.state.dispatched_job_count < number_of_jobs

        timespan = self.environment.now - shop_floor.history.started_at

        return timespan < shop_floor.configuration.configuration.timespan

    @staticmethod
    def from_cli(parameters, problem: Configuration, environment: simpy.Environment):
        return ShopFloorManager(
            environment=environment,
            job_sampler=job_sampler_from_cli(parameters['job_sampler'], problem=problem),
            breakdown=breakdown_from_cli(parameters['breakdown']),
            initial_job_assignment=initial_job_assignment_from_cli(parameters['initial_job_assignment']),
            seed=parameters['seed']
        )
