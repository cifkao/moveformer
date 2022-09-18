import datetime
import math
import os
import pickle
from typing import Callable, Dict, Optional, Union

import gps2var
import pyarrow as pa
import pyarrow.parquet as pq
import pytorch_lightning as pl
import jsonargparse
import numba
import numpy as np
import torch
from torch.utils.data import DataLoader

from .constants import PAD_INDEX
from .data_loading import DataProcessor
from .geo_utils import (
    gcs_to_distance,
    gcs_to_bearing,
    gcs_to_n_vector,
    apply_move,
    to_rad,
)
from .num_utils import find_repeated, split_ranges
from .training_utils import collate_sequence_dicts


Trajectory = Dict[str, np.ndarray]


def _traj_len(traj: Trajectory):
    return len(traj["timestamp"])


def augment_trajectory(
    traj: Trajectory, rng: np.random.Generator, drop_prob=0.0
) -> Trajectory:
    mask = rng.random(_traj_len(traj)) > drop_prob
    traj = {key: val[mask] for key, val in traj.items()}
    return traj


def add_n_vectors(traj: Trajectory, suffix: str = "") -> Trajectory:
    traj = dict(traj)  # Shallow copy
    if "location_lat" + suffix in traj:
        x, y, z = gcs_to_n_vector(
            traj["location_lat" + suffix], traj["location_long" + suffix]
        )
        traj["location_nvec" + suffix] = np.stack([x, y, z], axis=-1)
    return traj


PAD = "pad"
START = "start"
BEARING = "bearing"
DISTANCE = "distance"

DAY_SECONDS = 24 * 60 * 60
TROPICAL_YEAR_DAYS = 365.24219


def _encode_datetime(ts: np.datetime64, longitude) -> Dict[str, np.ndarray]:
    ts = ts.item()  # type: datetime.datetime
    sec_frac = ts.microsecond / 1e6
    min_frac = (ts.second + sec_frac) / 60
    hour_frac = (ts.minute + min_frac) / 60
    day_frac = (ts.hour + hour_frac) / 24

    ref_year_start = datetime.datetime(year=2000, month=1, day=1)
    years = (ts - ref_year_start).total_seconds() / DAY_SECONDS / TROPICAL_YEAR_DAYS

    enc_cont = np.array([sec_frac, min_frac, hour_frac, day_frac, years])
    enc_cont *= 2 * math.pi
    enc_cont = np.concatenate([np.sin(enc_cont), np.cos(enc_cont)])

    # Local mean time (LMT)
    lmt = (day_frac + longitude / 360) * 2 * math.pi
    lmt = np.array([np.sin(lmt), np.cos(lmt)])

    return {
        "time_cont": enc_cont,
        "time_month": ts.month - 1,
        "time_day": ts.day - 1,
        "time_weekday": ts.weekday(),
        "time_lmt": lmt,
    }


def _encode_timedelta(
    ts1: np.datetime64, ts2: np.datetime64, max_years=25
) -> Dict[str, np.ndarray]:
    dt = ts2.item() - ts1.item()  # type: datetime.timedelta
    sec_frac = dt.microseconds / 1e6
    day_frac = (dt.seconds + sec_frac) / DAY_SECONDS
    hour_frac = (day_frac * 24) % 1.0
    min_frac = (hour_frac * 60) % 1.0
    days = dt.days + day_frac
    years = days / TROPICAL_YEAR_DAYS

    enc_cont = np.array(
        [sec_frac, min_frac, hour_frac, day_frac, years, years / max_years]
    )
    enc_cont *= 2 * math.pi
    enc_cont = np.concatenate([np.sin(enc_cont), np.cos(enc_cont)])

    return enc_cont


@numba.njit
def _to_movement_vector(
    lat1: np.ndarray, long1: np.ndarray, lat2: np.ndarray, long2: np.ndarray
) -> np.ndarray:
    bearing = to_rad(gcs_to_bearing(lat1, long1, lat2, long2))
    distance = np.expand_dims(gcs_to_distance(lat1, long1, lat2, long2), axis=-1)
    return np.stack((np.sin(bearing), np.cos(bearing)), axis=-1) * distance


class SimpleTrajectoryProcessor(DataProcessor):
    def __init__(
        self,
        use_target=False,
        use_candidates=False,
        add_dtime=False,
        add_mov_vec=False,
        mov_vec_scale=1.0,
        handle_nan=False,
        pass_through=None,
    ):
        super().__init__()

        self._key_suffixes = [""]
        if use_target:
            self._key_suffixes.append("_tgt")
        if use_candidates:
            self._key_suffixes.append("_cand")
        self.add_dtime = add_dtime
        self.add_mov_vec = add_mov_vec
        self.mov_vec_scale = mov_vec_scale
        self.handle_nan = handle_nan
        self.pass_through = pass_through or []

    def encode(self, traj: Trajectory) -> Dict[str, np.ndarray]:
        for suf in self._key_suffixes:
            traj = add_n_vectors(traj, suf)
        traj_len = _traj_len(traj)

        encoded = {
            "mask": np.ones((traj_len,), dtype=np.bool_),
        }
        if self.handle_nan:
            encoded["loss_mask"] = encoded["mask"].copy()

        if self.add_dtime:
            encoded["dtime_cont"] = np.zeros((traj_len, 12), dtype=np.float32)
        for suf in self._key_suffixes:
            if "timestamp" + suf in traj:
                encoded.update(
                    {
                        "time_cont" + suf: np.zeros((traj_len, 10), dtype=np.float32),
                        "time_month" + suf: np.zeros((traj_len,), dtype=np.int64),
                        "time_day" + suf: np.zeros((traj_len,), dtype=np.int64),
                        "time_weekday" + suf: np.zeros((traj_len,), dtype=np.int64),
                        "time_lmt" + suf: np.zeros((traj_len, 2), dtype=np.float32),
                    }
                )

                for i in range(traj_len):
                    # UTC time + local mean time
                    for key, val in _encode_datetime(
                        traj["timestamp" + suf][i], traj["location_long" + suf][i]
                    ).items():
                        encoded[key + suf][i] = val

            # Location as n-vector
            if "location_nvec" + suf in traj:
                encoded["location" + suf] = traj["location_nvec" + suf].astype(
                    np.float32
                )

        # Delta time
        if self.add_dtime:
            for i in range(traj_len):
                encoded["dtime_cont"][i] = _encode_timedelta(
                    traj["timestamp"][i], traj["timestamp_tgt"][i]
                )

        # Movement vector
        if self.add_mov_vec:
            encoded["location_mov"] = np.zeros((traj_len, 2), dtype=np.float32)
            (valid_idxs,) = np.nonzero(~np.isnan(traj["location_lat"]))
            encoded["location_mov"][valid_idxs[1:]] = (
                _to_movement_vector(
                    traj["location_lat"][valid_idxs[:-1]],
                    traj["location_long"][valid_idxs[:-1]],
                    traj["location_lat"][valid_idxs[1:]],
                    traj["location_long"][valid_idxs[1:]],
                )
                * self.mov_vec_scale
            )

            if "location_tgt" in encoded:
                encoded["location_tgt_mov"] = (
                    _to_movement_vector(
                        traj["location_lat"],
                        traj["location_long"],
                        traj["location_lat_tgt"],
                        traj["location_long_tgt"],
                    ).astype(np.float32)
                    * self.mov_vec_scale
                )

            if "location_cand" in encoded:
                encoded["location_cand_mov"] = (
                    _to_movement_vector(
                        traj["location_lat"][:, None],
                        traj["location_long"][:, None],
                        traj["location_lat_cand"],
                        traj["location_long_cand"],
                    ).astype(np.float32)
                    * self.mov_vec_scale
                )

        if self.handle_nan:
            for suf in self._key_suffixes + ["_mov", "_tgt_mov", "_cand_mov"]:
                if "location" + suf not in encoded:
                    continue
                nan_mask = np.isnan(encoded["location" + suf])
                while nan_mask.ndim > 1:
                    nan_mask = nan_mask.any(axis=-1)
                encoded["location" + suf][nan_mask] = 0

                if suf in ["_tgt", "_cand"]:
                    encoded["loss_mask"] &= ~nan_mask

        for key in self.pass_through:
            for suf in self._key_suffixes:
                if key + suf not in traj:
                    continue
                encoded[key + suf] = traj[key + suf]
                if encoded[key + suf].dtype.kind == "f":
                    encoded[key + suf] = encoded[key + suf].astype(np.float32)
                elif encoded[key + suf].dtype.kind in "iu":
                    encoded[key + suf] = encoded[key + suf].astype(np.int64)
        return encoded


class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: Union[str, os.PathLike],
        preprocess_fn: Callable[[Trajectory], Dict[str, np.ndarray]] = lambda x: x,
        section: Optional[str] = None,
        split_max_len: Optional[int] = None,
        split_min_len: int = 1,
        encoded_max_len: Optional[int] = None,
        drop_prob: float = 0.0,
        target_max_skip: Optional[int] = None,
        deterministic: bool = False,
        id_column: str = "deployment_id",
        num_candidates: Optional[int] = None,
        candidate_sampler_path: Optional[Union[str, os.PathLike]] = None,
        geo_var_readers: Optional[Dict[str, gps2var.RasterValueReaderBase]] = None,
        var_joins: Optional[Dict[str, Dict[str, Union[str, os.PathLike]]]] = None,
    ):
        columns = [
            "timestamp",
            "location_lat",
            "location_long",
            id_column,
            "section",
        ]

        self.var_joiners = {}
        if var_joins:
            for var1, joins in var_joins.items():
                columns.append(var1)
                self.var_joiners[var1] = {}
                for var2, pkl_path in joins.items():
                    with open(pkl_path, "rb") as f:
                        self.var_joiners[var1][var2] = pickle.load(f)

        self.table = pq.read_table(
            path,
            columns=columns,
            filters=[("section", "==", section)] if section else None,
            memory_map=True,
        )

        # Build an index of all trajectories, assuming they form contiguous slices
        deployment_ids = self.table[id_column].to_numpy()
        if self.table.schema.field(id_column).type == pa.string():
            deployment_ids = deployment_ids.astype(str)
        _, self._index_offsets, index_ends = find_repeated(deployment_ids)
        if split_max_len is not None:
            self._index_offsets, index_ends, _ = split_ranges(
                self._index_offsets,
                index_ends,
                max_len=split_max_len,
                min_len=split_min_len,
            )
        self._index_lengths = index_ends - self._index_offsets

        self.candidate_sampler = None
        if candidate_sampler_path:
            with open(candidate_sampler_path, "rb") as f:
                self.candidate_sampler = pickle.load(f)

        self.preprocess_fn = preprocess_fn
        self.drop_prob = drop_prob
        self.encoded_max_len = encoded_max_len
        self.target_max_skip = target_max_skip
        self.id_column = id_column
        self.num_candidates = num_candidates
        self.geo_var_readers = geo_var_readers or {}
        self.rng = (
            None if deterministic else np.random.default_rng(np.random.randint(2**16))
        )

    def __len__(self):
        return len(self._index_offsets)

    def get_raw(self, idx):
        traj_pa = self.table.slice(self._index_offsets[idx], self._index_lengths[idx])
        traj = {col: traj_pa[col].to_numpy() for col in traj_pa.column_names}
        traj_id = traj_pa[self.id_column].to_numpy()[0]
        return traj, traj_id

    def __getitem__(self, idx):
        rng = self.rng or np.random.default_rng(idx)

        traj, traj_id = self.get_raw(idx)

        return self.encode(traj=traj, traj_id=traj_id, segment_id=idx, rng=rng)

    def encode(self, traj, traj_id, segment_id, tgt_as_cand=True, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        traj = dict(traj)  # Shallow copy

        if self.target_max_skip is not None and "location_lat_tgt" not in traj:
            # Load segments from the same trajectory that we can pick targets from
            future = [traj]
            for i in range(segment_id + 1, len(self)):
                traj2, traj2_id = self.get_raw(i)
                if traj2_id != traj_id:
                    break
                future.append(traj2)
                future_len = sum(_traj_len(tr) for tr in future[1:])
                if future_len >= self.target_max_skip + 1:
                    break
            future = {
                key: np.concatenate([tr[key] for tr in future]) for key in traj.keys()
            }
            if _traj_len(future) == _traj_len(traj):
                # If this is the last segment of this trajectory, remove the last point
                # to make sure we can always pick a target
                traj = {k: v[:-1] for k, v in traj.items()}

            # Sample targets at random from the future and add them
            target_idx = rng.integers(
                low=np.arange(_traj_len(traj)) + 1,
                high=(np.arange(_traj_len(traj)) + 1 + self.target_max_skip).clip(
                    max=_traj_len(future)
                ),
            )
            traj.update({f"{key}_tgt": val[target_idx] for key, val in future.items()})

        if "location_lat_tgt" not in traj:
            tgt_as_cand = False

        if self.num_candidates is not None and "location_lat_cand" not in traj:
            candidate_sampler = self.candidate_sampler
            if isinstance(candidate_sampler, dict):
                candidate_sampler = candidate_sampler[traj_id]
            num_samples = self.num_candidates
            if tgt_as_cand:
                num_samples -= 1

            (
                traj["location_lat_cand"],
                traj["location_long_cand"],
            ) = candidate_sampler.sample(traj=traj, num_samples=num_samples, rng=rng)

            if tgt_as_cand:
                # Add the target as the first candidate
                for coord in ["lat", "long"]:
                    traj[f"location_{coord}_cand"] = np.concatenate(
                        [
                            traj[f"location_{coord}_tgt"][..., None],
                            traj[f"location_{coord}_cand"],
                        ],
                        axis=-1,
                    )

        traj = augment_trajectory(traj, rng, drop_prob=self.drop_prob)

        for key, reader in self.geo_var_readers.items():
            for suf in ["", "_tgt", "_cand"]:
                if "location_lat" + suf in traj:
                    traj[key + suf] = reader.get(
                        traj["location_long" + suf], traj["location_lat" + suf]
                    )
                    if traj[key + suf].dtype.kind in "iu":
                        if traj[key + suf].shape[-1] == 1:
                            traj[key + suf] = traj[key + suf].squeeze(-1)

        for key, joiners in self.var_joiners.items():
            is_seq_level = (traj[key][0] == traj[key]).all()
            for var, joiner in joiners.items():
                assert var not in traj
                if is_seq_level:
                    # Sequence-level variable - retrieve only once and repeat
                    traj[var] = np.repeat(
                        np.array([joiner[traj[key][0]]]), _traj_len(traj), axis=0
                    )
                else:
                    traj[var] = np.array([joiner[x] for x in traj[key]])

        encoded = self.preprocess_fn(traj)

        if self.encoded_max_len is not None:
            length = min(len(x) for x in encoded.values())
            trim_length = max(0, length - self.encoded_max_len)
            # Some of the sequences may be 1 item longer. Need to reduce all by the same amount
            for key in encoded:
                encoded[key] = encoded[key][: len(encoded[key]) - trim_length]

        return encoded


class TrajectoryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: Union[str, os.PathLike],
        processor: DataProcessor,
        geo_var_readers: Optional[Dict[str, gps2var.RasterValueReaderBase]] = None,
        var_joins: Optional[Dict[str, Dict[str, Union[str, os.PathLike]]]] = None,
        **config,
    ):
        super().__init__()
        self.data_path = data_path
        self.datasets = {}
        self.processor = processor
        self.geo_var_readers = geo_var_readers
        self.var_joins = var_joins
        self.config = jsonargparse.Namespace(config)

    def setup(self, stage: Optional[str]):
        sections = ["val", "test"]
        if stage == "fit" or stage is None:
            sections.append("train")

        dataset_kwargs = {}
        if self.geo_var_readers:
            dataset_kwargs.update(geo_var_readers=self.geo_var_readers)
        if self.var_joins:
            dataset_kwargs.update(var_joins=self.var_joins)

        for section in sections:
            self.datasets[section] = TrajectoryDataset(
                path=self.data_path,
                preprocess_fn=self.processor.encode,
                section=section,
                **dataset_kwargs,
                **self.config[section],
            )

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            collate_fn=collate_sequence_dicts,
            **self.config["train_dataloader"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            collate_fn=collate_sequence_dicts,
            **self.config["infer_dataloader"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            collate_fn=collate_sequence_dicts,
            **self.config["infer_dataloader"],
        )
