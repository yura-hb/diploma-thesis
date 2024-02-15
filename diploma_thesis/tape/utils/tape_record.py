
from dataclasses import dataclass
from typing import TypeVar

from agents.utils.memory import Record


Context = TypeVar('Context')


@dataclass
class TapeRecord:
    moment: int
    record: Record
    context: Context
