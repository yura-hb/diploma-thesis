from .reward import *


class No(MachineReward):

    @dataclass
    class Context:
        pass

    @abstractmethod
    def record_job_action(self, job: Job, machine: Machine) -> Context:
        return self.Context()

    def reward_after_production(self, context: Context) -> torch.FloatTensor | None:
        """
        Returns: A tensor for
        """
        return None

    def reward_after_completion(self, contexts: List[Context]) -> torch.FloatTensor | None:
        return None

    @staticmethod
    def from_cli(parameters) -> MachineReward:
        return No()
