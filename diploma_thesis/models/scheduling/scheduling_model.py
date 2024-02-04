

import environment
import scheduling_rules


class SchedulingModel(environment.ShopFloor.Delegate, scheduling_rules.SchedulingRule):

    ...