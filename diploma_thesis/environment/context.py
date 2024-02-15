
import environment
from dataclasses import dataclass


@dataclass
class Context:
    shop_floor: 'environment.ShopFloor'
    moment: float
