from typing import Callable, Optional, Sequence, Union

import torch
from ignite.engine import Engine, Events
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = [
    'Running',
]

class Running(Metric):
    
    _required_output_keys = None

    def __init__(
        self,
        src: Optional[Metric] = None,
        output_transform: Optional[Callable] = None,
        epoch_bound: bool = True,
        reset_interval: Optional[Union[int, None]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if not (isinstance(src, Metric) or src is None):
            raise TypeError("Argument src should be a Metric or None.")

        if isinstance(src, Metric):
            if output_transform is not None:
                raise ValueError("Argument output_transform should be None if src is a Metric.")
            if device is not None:
                raise ValueError("Argument device should be None if src is a Metric.")
            self.src = src
            self._get_src_value = self._get_metric_value
            self.iteration_completed = self._metric_iteration_completed
        else:
            if output_transform is None:
                raise ValueError(
                    "Argument output_transform should not be None if src corresponds "
                    "to the output of process function."
                )
            self._get_src_value = self._get_output_value
            self.update = self._output_update

        self.epoch_bound = epoch_bound
        self.reset_interval = reset_interval
        super(Running, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self.src.reset()

    @reinit__is_reduced
    def update(self, output: Sequence) -> None:
        # Implement abstract method
        pass

    def compute(self) -> Union[torch.Tensor, float]:
        return self._get_metric_value()

    def attach(self, engine: Engine, name: str):
        if self.epoch_bound:
            # restart average every epoch
            engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        if self.reset_interval:
            engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.reset_interval), self.started)
        # compute metric
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        # apply running average
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.completed, name)

    def _get_metric_value(self) -> Union[torch.Tensor, float]:
        return self.src.compute()

    @sync_all_reduce("src")
    def _get_output_value(self) -> Metric:
        return self.src

    def _metric_iteration_completed(self, engine: Engine) -> None:
        self.src.iteration_completed(engine)

    @reinit__is_reduced
    def _output_update(self, output: Metric) -> None:
        self.src = output