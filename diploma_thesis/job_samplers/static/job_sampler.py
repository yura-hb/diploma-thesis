
import simpy

from environment import Job, Configuration, JobSampler


class Sampler(JobSampler):

    def __init__(self, problem: Configuration, environment: simpy.Environment):
        super().__init__(problem, environment)

        self._number_of_jobs = 0
        self._paths = None
        self._processing_times = None
        self._arrival_times = None
        self._due_times = None

        self._last_created_job_id = 0

    def number_of_jobs(self):
        return self._number_of_jobs

    def sample(self, job_id: int, initial_work_center_idx: int, moment: float) -> Job:
        self._last_created_job_id = job_id

        path = self._paths[job_id]
        processing_times = self._processing_times[job_id]
        due_at = self._due_times[job_id] + moment

        job = Job(id=job_id, step_idx=path, processing_times=processing_times)

        event = Job.Event(kind=Job.Event.Kind.creation, moment=moment)
        job = job.with_event(event)

        return job.with_due_at(due_at)

    def sample_next_arrival_time(self) -> float:
        return self._arrival_times[self._last_created_job_id]
