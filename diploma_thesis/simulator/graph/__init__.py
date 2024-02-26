
from .graph_model import GraphModel
from simulator.graph.encoder.unit_graph_encoder import UnitGraphEncoder
from simulator.graph.encoder.disjunctive_graph_encoder import DisjunctiveGraphEncoder
from simulator.graph.encoder.batch_disjunctive_graph_encoder import BatchGraphEncoder

from utils import from_cli
from functools import partial

key_to_class = {
    'unit': UnitGraphEncoder,
    'disjunctive': DisjunctiveGraphEncoder,
    'batch': BatchGraphEncoder
}

from_cli = partial(from_cli, key_to_class=key_to_class)
