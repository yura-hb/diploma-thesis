from .routing_rule import *


class ETRoutingRule(RoutingRule):

    @property
    def selector(self):
        return torch.argmin

    def criterion(self, job: Job, work_center: WorkCenter) -> torch.FloatTensor:
        return job.operation_processing_time_in_work_center(work_center.work_center_idx, JobReductionStrategy.none)