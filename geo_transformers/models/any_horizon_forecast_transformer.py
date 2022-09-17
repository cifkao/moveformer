from typing import Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.utilities.cli as pl_cli
import torch
from torch import nn
import torch.multiprocessing
from torch.nn import functional as F
from x_transformers.x_transformers import AttentionLayers, Attention

from ..nn import SequentialCVAE, masked_mse_loss
from ..constants import PAD_INDEX
from ..cli import TransformerCLI
from ..trajectory import (
    DAY_SECONDS,
    TROPICAL_YEAR_DAYS,
    TrajectoryDataset,
    TrajectoryDataModule,
)


class PredictionHead(nn.Module):
    def training_step(self, features, embeddings, dec_out, module: pl.LightningModule):
        _, loss, metrics = self.forward(features, embeddings, dec_out)
        self._log_metrics(metrics, "train", module)
        return loss

    def validation_step(
        self, features, embeddings, dec_out, module: pl.LightningModule
    ):
        _, loss, metrics = self.forward(features, embeddings, dec_out)
        self._log_metrics(metrics, "val", module)
        return loss

    def _log_metrics(self, metrics, prefix, module: pl.LightningModule):
        for name, value in metrics.items():
            module.log(f"{prefix}/{name}", value)


class AnyHorizonForecastTransformer(pl.LightningModule):
    def __init__(
        self,
        *,
        encoder: AttentionLayers,
        decoder: Optional[AttentionLayers] = None,
        prediction_head: PredictionHead,
        disc_feat_dims: Dict[str, int],
        cont_feat_dims: Dict[str, int],
        keys_in: List[str],
        keys_future: Optional[List[str]] = None,
        emb_dim: Optional[int] = None,
        learnable_nan: bool = False,
        concat_enc_to_out: bool = False,
        var_len_training: bool = False,
        var_len_training_v2: bool = False,
        max_len: Optional[int] = None,
        optimizer_init: Optional[dict] = None,
        lr_warmup_init: Optional[dict] = None,
        lr_decay_init: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.prediction_head = prediction_head
        self.keys_in = keys_in
        self.keys_future = keys_future
        self.learnable_nan = learnable_nan
        self.concat_enc_to_out = concat_enc_to_out
        self.var_len_training = var_len_training
        self.var_len_training_v2 = var_len_training_v2
        self.max_len = max_len
        self.optimizer_init = optimizer_init
        self.lr_warmup_init = lr_warmup_init
        self.lr_decay_init = lr_decay_init

        assert all(
            ll.causal for l in encoder.layers for ll in l if isinstance(ll, Attention)
        )
        if decoder:
            assert all(
                ll.causal
                for l in decoder.layers
                for ll in l
                if isinstance(ll, Attention)
            )

        if self.learnable_nan:
            self.nan_fill = nn.ParameterDict(
                {k: nn.Parameter(torch.zeros((d,))) for k, d in cont_feat_dims.items()}
            )

        emb_dim = emb_dim or encoder.dim
        self.disc_emb = nn.ModuleDict(
            {k: nn.Embedding(d, emb_dim) for k, d in disc_feat_dims.items()}
        )
        self.cont_proj = nn.ModuleDict(
            {k: nn.Linear(d, emb_dim) for k, d in cont_feat_dims.items()}
        )

        self.combine_in = nn.Linear(len(keys_in), 1)
        self.emb_proj = (
            nn.Linear(emb_dim, encoder.dim) if emb_dim != encoder.dim else nn.Identity()
        )
        self.encoder_norm = nn.LayerNorm(encoder.dim)
        if decoder:
            self.combine_future = nn.Linear(len(keys_future), 1)
            self.decoder_q_proj = nn.Linear(encoder.dim + emb_dim, decoder.dim)
            self.decoder_norm = nn.LayerNorm(decoder.dim)

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        run_prediction: bool = True,
        encoder_kwargs: Optional[dict] = None,
        decoder_kwargs: Optional[dict] = None,
    ):
        features = dict(features)
        mask = features["mask"]

        if self.learnable_nan:
            for key, fill_value in self.nan_fill.items():
                value = features[key]
                features[key] = torch.where(torch.isfinite(value), value, fill_value)

        embeddings = {
            k: embed(features[k])
            for k, embed in {**self.disc_emb, **self.cont_proj}.items()
            if k in features
        }

        # Transformer encoder
        emb_in = self.combine_in(
            torch.stack([embeddings[k] for k in self.keys_in], dim=-1)
        ).squeeze(-1)
        emb_in = self.emb_proj(emb_in)
        h = self.encoder(emb_in, mask=mask, **(encoder_kwargs or {}))
        h = self.encoder_norm(h)

        # Transformer decoder with (causal) cross-attention only
        #   query: [current encoder output, future features]
        #   keys: encoder outputs up to the current one
        if self.decoder:
            emb_future = self.combine_future(
                torch.stack([embeddings[k] for k in self.keys_future], dim=-1)
            ).squeeze(-1)
            dec_q = self.decoder_q_proj(torch.cat([h, emb_future], dim=-1))
            dec_out = self.decoder(
                dec_q, mask=mask, context=h, context_mask=mask, **(decoder_kwargs or {})
            )
            dec_out = self.decoder_norm(dec_out)

            if self.concat_enc_to_out:
                dec_out = torch.cat([dec_out, h], dim=-1)
        else:
            dec_out = h

        if run_prediction:
            return self.prediction_head(features, embeddings, dec_out)
        return embeddings, dec_out

    def configure_optimizers(self):
        optimizer = pl_cli.instantiate_class(self.parameters(), self.optimizer_init)
        schedulers = [
            {
                "scheduler": pl_cli.instantiate_class(optimizer, self.lr_decay_init),
                "interval": "epoch",
                "frequency": 1,
            },
            {
                "scheduler": pl_cli.instantiate_class(optimizer, self.lr_warmup_init),
                "interval": "step",
                "frequency": 1,
            },
        ]
        return [optimizer], schedulers

    def training_step(self, batch):
        self._log_length("train/length", batch)
        self._log_dtime("train/dtime_sec", batch)

        seq_len = batch["mask"].shape[1]
        max_len = self.max_len or seq_len
        attn_mask = None
        if self.var_len_training:
            # Partition the attention into two isolated blocks of random size
            split_idx = torch.randint(
                low=1, high=max_len + 1, size=(), device=self.device
            )
            if split_idx < seq_len:
                attn_mask = torch.zeros(
                    (seq_len, seq_len), dtype=torch.bool, device=self.device
                )
                attn_mask[:split_idx, :split_idx] = True
                attn_mask[split_idx:, split_idx:] = True
        elif self.var_len_training_v2:
            # Block-diagonal attention matrix with random block size
            ctx_len = torch.randint(
                low=1, high=max_len + 1, size=(), device=self.device
            )
            n_blocks = (seq_len + ctx_len - 1) // ctx_len
            attn_mask = torch.eye(n_blocks, dtype=torch.bool, device=self.device)
            attn_mask = attn_mask.repeat_interleave(ctx_len, dim=0)
            attn_mask = attn_mask.repeat_interleave(ctx_len, dim=1)
            attn_mask = attn_mask[:seq_len, :seq_len]

        embeddings, dec_out = self(
            batch,
            run_prediction=False,
            encoder_kwargs=dict(attn_mask=attn_mask),
            decoder_kwargs=dict(attn_mask=attn_mask),
        )
        loss = self.prediction_head.training_step(batch, embeddings, dec_out, self)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self._log_length("val/length", batch)
        self._log_dtime("val/dtime_sec", batch)
        embeddings, dec_out = self(batch, run_prediction=False)
        loss = self.prediction_head.validation_step(batch, embeddings, dec_out, self)
        self.log("val/loss", loss)
        return loss

    def _log_length(self, name, batch):
        self.log(name, batch["mask"].sum(dim=1).to(float).mean())

    def _log_dtime(self, name, batch):
        if "dtime_cont" in batch:
            sin, cos = (
                batch["dtime_cont"][:, :, [5, 10]].permute(2, 0, 1).to(torch.float64)
            )
            years = torch.atan2(sin, cos) / (2 * torch.pi) * 25
            seconds = years * TROPICAL_YEAR_DAYS * DAY_SECONDS
            mask = batch["mask"]
            self.log(name, (seconds * mask).sum() / mask.sum())


class LinearClassificationHead(PredictionHead):
    def __init__(self, num_features: int, num_classes: int, tgt_key: str):
        super().__init__()
        self.projection = nn.Linear(num_features, num_classes)
        self.tgt_key = tgt_key

    def forward(self, features, embeddings, dec_out):
        logits = self.projection(dec_out)
        loss = F.cross_entropy(
            logits.transpose(1, 2), features[self.tgt_key], ignore_index=PAD_INDEX
        )
        return logits, loss, {}


class LinearRegressionHead(PredictionHead):
    def __init__(self, in_features: int, out_features: int, tgt_key: str):
        super().__init__()
        self.projection = nn.Linear(in_features, out_features)
        self.tgt_key = tgt_key

    def forward(self, features, embeddings, dec_out):
        pred = self.projection(dec_out)
        loss = masked_mse_loss(pred, features[self.tgt_key], features["mask"])
        return pred, loss, {}


class CVAEHead(PredictionHead):
    def __init__(
        self,
        cvae: SequentialCVAE,
        tgt_key: str,
        kl_weight: float = 1.0,
        kl_anneal_steps: int = 0,
    ):
        super().__init__()
        self.cvae = cvae
        self.kl_weight = kl_weight
        self.kl_anneal_steps = kl_anneal_steps
        self.tgt_key = tgt_key

    def training_step(self, features, embeddings, dec_out, module: pl.LightningModule):
        _, (reco_loss, kl_loss) = self(dec_out, features, pred_mode=False)
        for var in ["reco_loss", "kl_loss"]:
            module.log(f"train/{var}", locals()[var])
        return self._get_total_loss(module, reco_loss, kl_loss)

    def validation_step(
        self, features, embeddings, dec_out, module: pl.LightningModule
    ):
        # Use the posterior (encoder) in order to compute reconstruction loss, ELBO
        _, (reco_loss, kl_loss) = self(dec_out, features, pred_mode=False)
        total_loss = self._get_total_loss(module, reco_loss, kl_loss)

        # Sample from the prior to compute prediction error
        _, (pred_loss, _) = self(features, embeddings, dec_out, pred_mode=True)

        for var in ["reco_loss", "kl_loss", "pred_loss"]:
            module.log(f"val/{var}", locals()[var])
        return total_loss

    def _get_total_loss(self, module: pl.LightningModule, reco_loss, kl_loss):
        kl_weight = self.kl_weight * min(
            1.0, (module.global_step + 1) / (self.kl_anneal_steps + 1)
        )
        if module.trainer.training:
            module.log("kl_weight", kl_weight)
        return reco_loss + kl_weight * kl_loss


class CVAEClassificationHead(CVAEHead):
    def forward(self, features, embeddings, dec_out, pred_mode=True):
        logits, _, kl_loss = self.cvae(
            dec_out, features[self.tgt_key], features["mask"], pred_mode=pred_mode
        )
        reco_loss = F.cross_entropy(
            logits.transpose(1, 2), features[self.tgt_key], ignore_index=PAD_INDEX
        )
        return logits, (reco_loss, kl_loss)


class CVAERegressionHead(CVAEHead):
    def forward(self, features, embeddings, dec_out, pred_mode=True):
        pred, _, kl_loss = self.cvae(
            dec_out, features[self.tgt_key], features["mask"], pred_mode=pred_mode
        )
        reco_loss = masked_mse_loss(pred, features[self.tgt_key], features["mask"])
        return pred, (reco_loss, kl_loss)


class SelectionHead(PredictionHead):
    def __init__(
        self,
        candidate_encoder_layers: List[nn.Module],
        embedding_in_keys: List[str],
        in_features: int,
        hidden_features: int,
    ):
        super().__init__()
        self.candidate_encoder = nn.Sequential(*candidate_encoder_layers)
        self.combine_emb = nn.Linear(len(embedding_in_keys), 1)
        self.query_proj = nn.Linear(in_features, hidden_features)
        self.embedding_in_keys = embedding_in_keys

    def forward(self, features, embeddings, dec_out):
        # Gather embeddings of candidate features, shape (batch, seq, cand, emb_dim)
        cand_emb = self.combine_emb(
            torch.stack([embeddings[k] for k in self.embedding_in_keys], dim=-1)
        ).squeeze(-1)
        # Concatenate with decoder output, expanded to match the number of candidates
        cand_emb = torch.cat(
            [cand_emb, dec_out.unsqueeze(-2).expand(*cand_emb.shape[:-1], -1)], dim=-1
        )
        # Encode everything
        cand_emb = self.candidate_encoder(cand_emb)

        query = self.query_proj(dec_out)
        scale = query.shape[-1] ** -0.5
        logits = torch.einsum("b n c d, b n d -> b n c", cand_emb, query) * scale

        # Assume first one is the target
        loss = F.cross_entropy(
            logits.transpose(1, 2),
            torch.zeros(logits.shape[:2], dtype=torch.int64, device=logits.device),
            reduction="none",
        )
        mask = features.get("loss_mask", features["mask"])
        loss = (loss * mask).sum() / mask.sum() / np.log(logits.shape[-1])

        # Measure top-1, top-5 and top-50% accuracy
        half = logits.shape[-1] // 2
        metrics = {
            "acc": ((logits.argmax(-1) == 0) * mask).sum() / mask.sum(),
            f"acc_top{half}": (
                ((logits.topk(half).indices == 0).any(-1) * mask).sum() / mask.sum()
            ),
        }
        if half != 5:
            metrics["acc_top5"] = (
                (logits.topk(5).indices == 0).any(-1) * mask
            ).sum() / mask.sum()

        return logits, loss, metrics


class AnyHorizonForecastTransformerCLI(TransformerCLI):
    def __init__(self, **kwargs):
        super().__init__(AnyHorizonForecastTransformer, TrajectoryDataModule, **kwargs)

    def add_arguments_to_parser(self, parser: pl_cli.LightningArgumentParser):
        super().add_arguments_to_parser(parser)
        for section in ["train", "val", "test"]:
            parser.add_class_arguments(
                TrajectoryDataset,
                f"data.{section}",
                instantiate=False,
                skip=[
                    "path",
                    "preprocess_fn",
                    "section",
                    "geo_var_readers",
                    "var_joins",
                ],
            )


if __name__ == "__main__":
    cli = AnyHorizonForecastTransformerCLI()
