from fvcore.common.param_scheduler import ParamScheduler
import math


class LinearWarmupCosineParamScheduler(ParamScheduler):
    """
    A cosine scheduler with warmup.
    """
    def __init__(
            self,
            start_value: float,
            end_value: float,
            warmup_duration: float = 0.0,
            warmup_rate: float = 100.0,
    ) -> None:
        self._start_value = start_value
        self._end_value = end_value
        self._warmup_duration = warmup_duration
        self._warmup_rate = warmup_rate

    def __call__(self, where: float) -> float:
        """
        Args:
            where (float): A float between 0 and 1. This value is used to
                determine the current step in the scheduler.
        """
        if where < self._warmup_duration:
            warmup_start = self._start_value / self._warmup_rate
            warmup_end = self._start_value
            this_lr = where / self._warmup_duration * (warmup_end - warmup_start) + warmup_start
            return this_lr
        else:
            decay_where = (where - self._warmup_duration) / (1 - self._warmup_duration)
            return self._end_value + 0.5 * (self._start_value - self._end_value) * (
                    1 + math.cos(math.pi * decay_where)
            )

