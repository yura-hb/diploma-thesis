from typing import Dict

from .atc import ATCSchedulingRule
from .avpro import AVPROSchedulingRule
from .covert import COVERTSchedulingRule
from .cr import CRSchedulingRule
from .crspt import CRSPTSchedulingRule
from .dptlwkr import DPTLWKRSchedulingRule
from .dptlwkrs import DPTLWKRSSchedulingRule
from .dptwinqnpt import DPTWINQNPTSchedulingRule
from .edd import EDDSchedulingRule
from .fifo import FIFOSchedulingRule
from .gp_1 import GP1SchedulingRule
from .gp_2 import GP2SchedulingRule
from .idle import IdleSchedulingRule
from .lifo import LIFOSchedulingRule
from .lpt import LPTSchedulingRule
from .lro import LROSchedulingRule
from .lwkr import LWRKSchedulingRule
from .lwt import LWTSchedulingRule
from .lwkrmod import LWRKMODSchedulingRule
from .lwkrspt import LWRKSPTSchedulingRule
from .mwkr import MWRKSchedulingRule
from .mdd import MDDSchedulingRule
from .mod import MODSchedulingRule
from .mon import MONSchedulingRule
from .mro import MROSchedulingRule
from .ms import MSSchedulingRule
from .npt import NPTSchedulingRule
from .ptwinq import PTWINQSchedulingRule
from .ptwinqs import PTWINQSSchedulingRule
from .random import RandomSchedulingRule
from .scheduling_rule import SchedulingRule
from .spmwk import SPMWKSchedulingRule
from .spmwkspt import SPMWKSPTSchedulingRule
from .spt import SPTSchedulingRule
from .swt import SWTSchedulingRule
from .winq import WINQSchedulingRule

ALL_SCHEDULING_RULES: Dict[str, SchedulingRule.__class__] = {
    'atc': ATCSchedulingRule,
    'avpro': AVPROSchedulingRule,
    'covert': COVERTSchedulingRule,
    'cr': CRSchedulingRule,
    'crspt': CRSPTSchedulingRule,
    'dptlwkr': DPTLWKRSchedulingRule,
    'dptlwkrs': DPTLWKRSSchedulingRule,
    'dptwinqnpt': DPTWINQNPTSchedulingRule,
    'edd': EDDSchedulingRule,
    'fifo': FIFOSchedulingRule,
    'gp1': GP1SchedulingRule,
    'gp2': GP2SchedulingRule,
    'lifo': LIFOSchedulingRule,
    'lpt': LPTSchedulingRule,
    'lro': LROSchedulingRule,
    'lwkr': LWRKSchedulingRule,
    'lwkrmod': LWRKMODSchedulingRule,
    'lwkrspt': LWRKSPTSchedulingRule,
    'lwt': LWTSchedulingRule,
    'mwkr': MWRKSchedulingRule,
    'mdd': MDDSchedulingRule,
    'mod': MODSchedulingRule,
    'mon': MONSchedulingRule,
    'mro': MROSchedulingRule,
    'ms': MSSchedulingRule,
    'npt': NPTSchedulingRule,
    'ptwinq': PTWINQSchedulingRule,
    'ptwinqs': PTWINQSSchedulingRule,
    'random': RandomSchedulingRule,
    'spmwk': SPMWKSchedulingRule,
    'spmwkspt': SPMWKSPTSchedulingRule,
    'spt': SPTSchedulingRule,
    'swt': SWTSchedulingRule,
    'winq': WINQSchedulingRule
}

def from_cli(parameters):
    rules = parameters['rules']

    all_rules = ALL_SCHEDULING_RULES

    if rules == "all":
        rules = [rule() for rule in all_rules.values()]
    else:
        rules = [all_rules[rule]() for rule in rules]

    if parameters.get('idle', False):
        rules = [IdleSchedulingRule()] + rules

    return rules