from .atc import ATCSchedulingRule
from .avpro import AVPROSchedulingRule
from .covert import COVERTSchedulingRule
from .cr import CRSchedulingRule
from .crspt import CRSPTSchedulingRule
from .edd import EDDSchedulingRule
from .fifo import FIFOSchedulingRule
from .lpt import LPTSchedulingRule
from .lro import LROSchedulingRule
from .lwkr import LWRKSchedulingRule
from .lwkrmod import LWRKMODSchedulingRule
from .mdd import MDDSchedulingRule
from .mod import MODSchedulingRule
from .mon import MONSchedulingRule
from .ms import MSSchedulingRule
from .npt import NPTSchedulingRule
from .random import RandomSchedulingRule
from .spmwk import SPMWKSchedulingRule
from .spmwkspt import SPMWKSPTSchedulingRule

ALL_SCHEDULING_RULES = {
    'atc': ATCSchedulingRule,
    'avpro': AVPROSchedulingRule,
    'covert': COVERTSchedulingRule,
    'cr': CRSchedulingRule,
    'crspt': CRSPTSchedulingRule,
    'edd': EDDSchedulingRule,
    'fifo': FIFOSchedulingRule,
    'lpt': LPTSchedulingRule,
    'lro': LROSchedulingRule,
    'lwrk': LWRKSchedulingRule,
    'lwrkmod': LWRKMODSchedulingRule,
    'mdd': MDDSchedulingRule,
    'mod': MODSchedulingRule,
    'mon': MONSchedulingRule,
    'ms': MSSchedulingRule,
    'npt': NPTSchedulingRule,
    'random': RandomSchedulingRule,
    'spmwk': SPMWKSchedulingRule,
    'spmwkspt': SPMWKSPTSchedulingRule,
}