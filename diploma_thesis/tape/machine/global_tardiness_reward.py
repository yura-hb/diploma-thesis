from .reward import *


class GlobalTardiness(MachineReward):
    """
    Reward from Deep-MARL external/PhD-Thesis-Projects/JSP/machine.py:693
    """

    @dataclass
    class Context:
        job: Job
        machine: Machine

    def record_job_action(self, job: Job, machine: Machine) -> Context:
        return self.Context(job, machine)

    def reward_after_production(self, context: Context) -> torch.FloatTensor | None:
        return None

    def reward_after_completion(self, context: List[Context]) -> torch.FloatTensor | None:
        pass
        # assert job.is_completed, f"Job {job} is not completed"
        #
        # reward = torch.zeros_like(job.step_idx, dtype=torch.float)
        #
        # if job.is_tardy_upon_completion:
        #     tardy_rate = - torch.clip(job.tardiness_upon_completion / 256, 0, 1)
        #
        #     reward += tardy_rate
        #
        # for index, work_center_idx in enumerate(job.step_idx):
        #     agent_reward = reward[index]
        #
        #     machine = shop_floor.machine(work_center_idx, job.history.arrived_at_machine[index])

    @staticmethod
    def from_cli(parameters) -> MachineReward:
        return GlobalTardiness()
