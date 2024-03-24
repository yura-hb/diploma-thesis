from functools import partial

from utils import from_cli
from .persisted import PersistedAgent
from .static import StaticCandidates
from .template import Template, Candidate

key_to_cls = {
    "static": StaticCandidates,
    "persisted_agents": PersistedAgent
}


from_cli = partial(from_cli, key_to_class=key_to_cls)