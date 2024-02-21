from .simulator import *


class EpisodicSimulator(Simulator):
    """
    A simulator, which launches several shop=floors in parallel and simulates them until terminating conditions are met.
    During the process of the run the whole episode of environment is recorded.

    After the run is finished returns are estimated and passed to the agent for training.
    """

    def did_prepare_machine_record(self,
                                   shop_floor: ShopFloor,
                                   machine: Machine,
                                   record: Record,
                                   decision_moment: float):
       pass

    def did_prepare_work_center_record(self,
                                       shop_floor: ShopFloor,
                                       work_center: WorkCenter,
                                       record: Record,
                                       decision_moment: float):
        pass

    @staticmethod
    def from_cli(parameters, *args, **kwargs) -> Simulator:
        return EpisodicSimulator(*args, **kwargs)