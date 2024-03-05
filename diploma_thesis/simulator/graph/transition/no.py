from .transition import *


class No(GraphTransition):

    def append(self, job: Job, shop_floor: ShopFloor, graph: Graph):
        pass

    def update_on_dispatch(self, job: Job, shop_floor: ShopFloor, graph: Graph):
        pass

    def update_on_will_produce(self, job: Job, shop_floor: ShopFloor, graph: Graph):
        pass

    def remove(self, job: Job, shop_floor: ShopFloor, graph: Graph):
        pass

    def flatten(self, graph: Graph):
        return None

    @classmethod
    def from_cli(cls, parameters: Dict):
        return cls(None, None)
