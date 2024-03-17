
from .estimator import Estimator as ReturnEstimator, ValueFetchMethod

from .no import No
from .gae import GAE
from .n_step import NStep

from utils import from_cli as _from_cli
from functools import partial

key_to_class = {
    'no': No,
    'gae': GAE,
    'n_step': NStep
}

from_cli = partial(_from_cli, key_to_class=key_to_class)
