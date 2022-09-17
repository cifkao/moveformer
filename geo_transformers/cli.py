import pytorch_lightning.utilities.cli as pl_cli
import torch
from torch.utils.data import DataLoader

from .training_utils import LinearWarmupLR


class TransformerCLI(pl_cli.LightningCLI):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            save_config_callback=None,
            **kwargs,
        )

    def instantiate_classes(self) -> None:
        cfg = self.config[self.subcommand]
        cfg.trainer.logger.init_args.config = self.config.as_dict()
        super().instantiate_classes()

    def add_arguments_to_parser(self, parser: pl_cli.LightningArgumentParser):
        for mode in ["train", "infer"]:
            parser.add_class_arguments(
                DataLoader,
                f"data.{mode}_dataloader",
                instantiate=False,
                skip=["dataset", "collate_fn"],
            )
        parser.set_defaults({"data.train_dataloader.shuffle": True})
        parser.add_optimizer_args(torch.optim.Adam, link_to="model.optimizer_init")
        parser.add_lr_scheduler_args(
            LinearWarmupLR, nested_key="lr_warmup", link_to="model.lr_warmup_init"
        )
        parser.add_lr_scheduler_args(
            torch.optim.lr_scheduler.StepLR,
            nested_key="lr_decay",
            link_to="model.lr_decay_init",
        )

    def before_instantiate_classes(self):
        config = self.config[self.subcommand]
        logger = config.trainer.logger
        # Workaround for PL bug
        del logger.init_args.agg_key_funcs, logger.init_args.agg_default_func
