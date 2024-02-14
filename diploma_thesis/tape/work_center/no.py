from .reward import *


class No(WorkCenterReward):
    """
    Reward from Deep-MARL external/PhD-Thesis-Projects/FJSP/agent_machine.py:803
    """

    @dataclass
    class Context:
        pass

    def record_job_action(self, job: Job, work_center: WorkCenter) -> Context:
        return self.Context()

    def reward_after_production(self, context: Context) -> torch.FloatTensor | None:
        pass

    def reward_after_completion(self, context: List[Context]) -> torch.FloatTensor | None:
        pass

    @staticmethod
    def from_cli(parameters: dict):
        return No()
