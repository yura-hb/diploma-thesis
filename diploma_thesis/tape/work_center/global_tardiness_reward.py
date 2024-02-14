from .reward import *


class GlobalTardiness(WorkCenterReward):
    """
    Reward from Deep-MARL external/PhD-Thesis-Projects/FJSP/agent_machine.py:803
    """

    @staticmethod
    def from_cli(parameters: dict):
        return GlobalTardiness()

    # def add_global_reward_RA(self):  # BASELINE RULE !!!
    #     job_record = self.job_creator.production_record[self.job_idx]
    #     path = job_record[1]
    #     queued_time = np.array(job_record[2])
    #     # if tardiness is non-zero and waiting time exists, machines in path get punishment
    #     if self.tardiness and queued_time.sum():
    #         global_reward = - np.clip(self.tardiness / 64, 0, 1)
    #         tape = torch.ones(len(queued_time), dtype=torch.float) * global_reward
    #     else:
    #         tape = torch.ones(len(queued_time), dtype=torch.float) * 0
    #     # print(queued_time)
    #     # print(self.tardiness,tape)
    #     for i, m_idx in enumerate(path):
    #         r_t = tape[i]
    #         wc_idx = self.m_list[m_idx].wc_idx
    #         try:
    #             self.wc_list[wc_idx].incomplete_experience[self.job_idx].insert(2, r_t)
    #             self.wc_list[wc_idx].rep_memo.append(self.wc_list[wc_idx].incomplete_experience.pop(self.job_idx))
    #         except:
    #             pass
