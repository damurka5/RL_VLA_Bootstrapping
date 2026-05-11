from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable, Sequence

import numpy as np


class PerOptionReplayBuffer:
    def __init__(self, capacity_per_option: int):
        self.capacity_per_option = max(1, int(capacity_per_option))
        self._buffers: dict[str, deque[Any]] = defaultdict(lambda: deque(maxlen=self.capacity_per_option))

    def add(self, option_name: str, record: Any) -> None:
        self._buffers[str(option_name)].append(record)

    def extend(self, option_name: str, records: Iterable[Any]) -> None:
        for record in records:
            self.add(option_name, record)

    def options(self) -> tuple[str, ...]:
        return tuple(sorted(option for option, items in self._buffers.items() if items))

    def sizes(self) -> dict[str, int]:
        return {option: len(items) for option, items in sorted(self._buffers.items())}

    def __len__(self) -> int:
        return sum(len(items) for items in self._buffers.values())

    def sample_balanced(
        self,
        *,
        batch_size: int,
        rng: np.random.Generator,
        allowed_options: Sequence[str] | None = None,
        option_weights: dict[str, float] | None = None,
    ) -> list[Any]:
        batch_size = max(0, int(batch_size))
        if batch_size == 0:
            return []

        if allowed_options is None:
            options = [option for option, items in self._buffers.items() if items]
        else:
            allowed = {str(option) for option in allowed_options}
            options = [option for option, items in self._buffers.items() if option in allowed and items]
        if not options:
            return []

        raw_weights = []
        for option in options:
            weight = 1.0
            if option_weights and option in option_weights:
                weight = max(0.0, float(option_weights[option]))
            raw_weights.append(weight)
        weights = np.asarray(raw_weights, dtype=np.float64)
        if not np.any(weights > 0.0):
            weights = np.ones_like(weights)
        weights = weights / float(weights.sum())

        out: list[Any] = []
        for _ in range(batch_size):
            option = str(rng.choice(options, p=weights))
            items = self._buffers[option]
            idx = int(rng.integers(0, len(items)))
            out.append(items[idx])
        return out
