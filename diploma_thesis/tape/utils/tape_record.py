
from dataclasses import dataclass
from typing import TypeVar

from agents.utils.memory import Record


Context = TypeVar('Context')


@dataclass
class TapeRecord:
    record: Record
    context: Context
    moment: int
