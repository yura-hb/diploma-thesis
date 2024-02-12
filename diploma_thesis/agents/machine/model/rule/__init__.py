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
from .lifo import LIFOSchedulingRule
from .lpt import LPTSchedulingRule
from .lro import LROSchedulingRule
from .lwkr import LWRKSchedulingRule
from .lwkrmod import LWRKMODSchedulingRule
from .lwkrspt import LWRKSPTSchedulingRule
from .mdd import MDDSchedulingRule
from .mod import MODSchedulingRule
from .mon import MONSchedulingRule
from .ms import MSSchedulingRule
from .npt import NPTSchedulingRule
from .ptwinq import PTWINQSchedulingRule
from .ptwinqs import PTWINQSSchedulingRule
from .random import RandomSchedulingRule
from .spmwk import SPMWKSchedulingRule
from .spmwkspt import SPMWKSPTSchedulingRule
from .winq import WINQSchedulingRule
from .scheduling_rule import SchedulingRule
from typing import Dict

ALL_SCHEDULING_RULES: Dict[str, SchedulingRule] = {
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
    'mdd': MDDSchedulingRule,
    'mod': MODSchedulingRule,
    'mon': MONSchedulingRule,
    'ms': MSSchedulingRule,
    'npt': NPTSchedulingRule,
    'ptwinq': PTWINQSchedulingRule,
    'ptwinqs': PTWINQSSchedulingRule,
    'random': RandomSchedulingRule,
    'spmwk': SPMWKSchedulingRule,
    'spmwkspt': SPMWKSPTSchedulingRule,
    'winq': WINQSchedulingRule
}
