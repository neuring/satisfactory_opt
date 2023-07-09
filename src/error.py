from dataclasses import dataclass
from typing import Optional, List

from source import Span

@dataclass
class Error(Exception):

    @property
    def span(self) -> List[Span]:
        raise NotImplementedError()

    def create_message(self) -> str:
        raise NotImplementedError()