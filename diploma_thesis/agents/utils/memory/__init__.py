
from functools import partial

from utils import from_cli
from .memory import Record, Memory, NotReadyException
from .prioritized_replay_memory import PrioritizedReplayMemory
from .replay_memory import ReplayMemory

key_to_cls = {
    'replay': ReplayMemory,
    'prioritized_replay': PrioritizedReplayMemory
}

from_cli = partial(from_cli, key_to_class=key_to_cls)
