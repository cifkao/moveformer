import contextlib
from multiprocessing.managers import BaseManager
from typing import Any, Callable, List, Optional, TypeVar, Union

import gps2var
from gps2var.core import RasterValueReaderBase, RasterReaderSpecLike
import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")


def collate_sequence_dicts(batch: List[dict]):
    return {
        key: pad_sequence([torch.as_tensor(x[key]) for x in batch], batch_first=True)
        for key in batch[0]
    }


class LinearWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """Chainable linear warmup scheduler."""

    def __init__(self, optimizer, warmup_steps, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        super(LinearWarmupLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch > self.warmup_steps:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [lr * self.last_epoch / self.warmup_steps for lr in self.base_lrs]


class WandbWatchCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        trainer.logger.watch(pl_module)

    def on_train_end(self, trainer, pl_module):
        trainer.logger.unwatch(pl_module)


# The following are wrappers for objects that spawn them in separate processes


class ProcessManagerWrapper:
    def __init__(self, cls: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> None:
        super().__init__()

        self._exit_stack = contextlib.ExitStack()

        Manager = type(f"Manager[{cls}]", (BaseManager,), {})  # type: BaseManager
        Manager.register(cls.__name__, cls)
        self._manager = self._exit_stack.enter_context(Manager())
        try:
            self._proxy = getattr(self._manager, cls.__name__)(*args, **kwargs)
        except:
            with self._exit_stack:
                raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return self._exit_stack.__exit__(exc_type, exc_value, traceback)

    def close(self):
        self.__exit__(None, None, None)

    def __del__(self):
        self.close()


class RasterValueReaderPool(ProcessManagerWrapper, RasterValueReaderBase):
    def __init__(
        self,
        spec: Union[RasterReaderSpecLike, List[RasterReaderSpecLike]],
        num_workers: int,
        num_threads: Optional[int] = None,
        use_multiprocessing: bool = False,
        **kwargs,
    ) -> None:
        if isinstance(spec, list):
            kwargs.update(
                num_threads=num_threads, use_multiprocessing=use_multiprocessing
            )
        super().__init__(
            gps2var.RasterValueReaderPool, spec=spec, num_workers=num_workers, **kwargs
        )
        self._exit_stack.enter_context(contextlib.closing(self._proxy))

    def get(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        extra: Any = None,
    ) -> np.ndarray:
        return self._proxy.get(x, y, extra)
