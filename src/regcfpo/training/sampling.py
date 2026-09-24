"""Deterministic cycle-exposure sampler for the AReG 348-step diagnostic.

Pure and CPU-testable (torch only).  Amendment
``asymmetric_directional_credit_revision_20260826`` exposure_rule: every
manifest row is visited EXACTLY once per cycle in ``cycle_position`` order.
The vendored ``RepeatRandomSampler`` draws a fresh ``randperm`` every epoch
(sampling with replacement across epochs: some rows repeat, some are never
seen), so it cannot provide the registered schedule and is replaced here via
the ``trainer_class`` instrumentation seam of ``scripts/train_stage4.py``.

The train dataset must already be sorted by ``cycle_position``
(``train_stage4.load_data_rows`` does this for merged manifests), so the
sequential layout below IS the cycle order.  One 348-row cycle at one prompt
group per optimizer step (per-device batch 4, G=4) is exactly 348 steps.
"""

from __future__ import annotations

from torch.utils.data import Sampler


class CycleOrderSampler(Sampler):
    """Sequential group-layout sampler: each row exactly once per cycle.

    Mirrors the vendored ``RepeatRandomSampler`` index layout (each index
    repeated ``mini_repeat_count`` times consecutively so consecutive
    ``num_generations`` entries form one rollout group, ``batch_size``
    unique indices per batch, ``repeat_count`` passes per ``__iter__``) but
    iterates the dataset in its given order instead of a seeded randperm.
    ``__len__`` matches the vendored formula so the Trainer's step/epoch
    arithmetic is unchanged.  Unlike the vendored sampler, an incomplete
    trailing batch chunk fails fast — silently dropping rows would violate
    the every-row-exactly-once contract.
    """

    def __init__(
        self,
        data_source,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
    ) -> None:
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        if self.num_samples == 0:
            raise ValueError("CycleOrderSampler requires a non-empty dataset")
        if mini_repeat_count < 1 or batch_size < 1 or repeat_count < 1:
            raise ValueError("mini_repeat_count, batch_size, repeat_count must be >= 1")
        if self.num_samples % self.batch_size != 0:
            raise ValueError(
                f"dataset size {self.num_samples} is not divisible by the "
                f"unique-index batch size {self.batch_size}; the cycle-exposure "
                "contract (every row exactly once) would be silently violated"
            )

    def __iter__(self):
        for start in range(0, self.num_samples, self.batch_size):
            chunk = range(start, start + self.batch_size)
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


__all__ = ["CycleOrderSampler"]
