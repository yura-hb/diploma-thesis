from .reward import *
from dataclasses import dataclass


# class SurrogateTardinessRewardModel(MachineRewardModel):
#     """
#     Reward from Deep-MARL external/PhD-Thesis-Projects/FJSP/agent_machine.py:753
#     """
#
#     @dataclass
#     class Record:
#         job_ids: List[int]
#
#     def __init__(self):
#         super().__init__()
#
#         self.queue: Dict[str, Dict[MachineKey, Dict[int, int]]] = dict()
#
#     def will_produce(self, shop_floor: ShopFloor, machine: Machine, job: Job):
#         self.queue[shop_floor.id][machine.key][job.id] = 1
#
#     def did_produce(self, shop_floor: ShopFloor, machine: Machine, job: Job):
#         record = self.queue[shop_floor.id][machine.key].get(job.id)
#
#         if record is None:
#             return
#
#         del self.queue[shop_floor.id][machine.key][job.id]
#
#     @staticmethod
#     def from_cli(parameters) -> MachineRewardModel:
#         return SurrogateTardinessRewardModel()

#
# def get_reward13(self):
#        slack = self.before_op_slack
#        critical_level = 1 - slack / (np.absolute(slack) + 64)
#        # get critical level for jobs, chosen and loser, respectively
#        critical_level_chosen = critical_level[self.position]
#        critical_level_loser = np.delete(critical_level, self.position) # could be a vector or scalar
#        # calculate adjusted earned slack for the chosen job
#        earned_slack_chosen = np.mean(self.current_pt[:self.waiting_jobs-1])
#        earned_slack_chosen *= critical_level_chosen
#        # calculate the AVERAGE adjusted slack consumption for jobs that not been chosen
#        consumed_slack_loser = self.pt_chosen*critical_level_loser.mean()
#        # slack tape
#        rwd_slack = earned_slack_chosen - consumed_slack_loser
#        # WINQ tape
#        rwd_winq = (self.before_op_winq_loser.mean() - self.before_op_winq_chosen) * 0.2
#        # calculate the tape
#        #print(rwd_slack, rwd_winq)
#        rwd = ((rwd_slack + rwd_winq)/20).clip(-1,1)
#        # optional printout
#        #print(self.env.now,'slack and pt:', slack, critical_level, self.position, self.pt_chosen, self.current_pt[:self.waiting_jobs-1])
#        #print(self.env.now,'winq and tape:',self.before_op_winq_chosen, self.before_op_winq_loser, earned_slack_chosen, consumed_slack_loser)
#        #print(self.env.now,'tape:',rwd)
#        r_t = torch.tensor(rwd , dtype=torch.float)
#        return r_t
