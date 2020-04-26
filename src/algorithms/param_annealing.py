import numpy as np
from typing import Optional


class AnnealedParam(float):

    def __new__(
            cls,
            param_min: float,
            param_max: float,
            period: int,
            param_value: Optional[float] = None,
            param_max_decay: float = 1.0,
            param_min_decay: float = 1.0,
            schedule_type: str = "linear",
            iteration: int = 0
    ):
        param_value = param_value if param_value else param_max
        return float.__new__(cls, param_value)

    def __getnewargs__(self):
        return self.param_min, self.param_max, self.period

    def __init__(
            self,
            param_min: float,
            param_max: float,
            period: int,
            param_value: Optional[float] = None,
            param_max_decay: float = 1.0,
            param_min_decay: float = 1.0,
            schedule_type: str = "linear",
            iteration: int = 0
    ):
        assert param_min <= param_max, ValueError
        param_value = param_value if param_value else param_max
        float.__init__(param_value)
        self.param_min = param_min
        self.param_max = param_max
        self.period = period
        self.param_max_decay = param_max_decay
        self.param_min_decay = param_min_decay
        self.schedule_type = schedule_type
        self.iteration = iteration

    def calculate_param_from_schedule(self) -> float:
        if self.schedule_type == "linear":
            cycle_pct = (self.period - self.iteration % self.period) / self.period
            return self.param_min + (self.param_max - self.param_min) * cycle_pct
        elif self.schedule_type == "sinusoidal":
            cycle_pct = (1 + np.cost(np.pi * self.iteration / self.period)) / 2
            return self.param_min + (self.param_max - self.param_min) * cycle_pct
        else:
            raise NotImplementedError

    def update(self) -> float:
        new_param_value = self.calculate_param_from_schedule()
        self.param_max = self.param_max_decay*self.param_max + (1 - self.param_max_decay)*self.param_min
        self.param_min = self.param_min_decay*self.param_min + (1 - self.param_min_decay)*self.param_max
        self.iteration += 1
        return AnnealedParam(**self.__dict__, param_value=new_param_value)

    def __str__(self):
        return f"annealed_{'{0:.1E}'.format(self.param_min)}_{'{0:.1E}'.format(self.param_max)}_{self.period}".replace(
            ".", "-"
        )
