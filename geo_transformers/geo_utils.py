"""Fast implementations of geography utility functions.

Assumes the spherical model of Earth.
"""

import math
from typing import Dict, Optional, Tuple

import numba
import numpy as np


EARTH_R = 6371008.7714  # mean radius from https://doi.org/10.1007/s001900050278


@numba.njit
def hav(x: np.ndarray) -> np.ndarray:
    """Haversine function."""
    return np.sin(0.5 * x) ** 2


@numba.njit
def to_rad(x: np.ndarray) -> np.ndarray:
    """Convert degrees to radians."""
    return x * math.pi / 180


@numba.njit
def to_deg(x: np.ndarray) -> np.ndarray:
    """Convert radians to degrees."""
    return x * 180 / math.pi


@numba.njit
def gcs_to_distance(
    lat1: np.ndarray, long1: np.ndarray, lat2: np.ndarray, long2: np.ndarray
) -> np.ndarray:
    """Compute the distance in meters between two points given by latitude and longitude
    in degrees.

    Formula from: https://en.wikipedia.org/wiki/Great-circle_distance
    """
    lat1, long1, lat2, long2 = to_rad(lat1), to_rad(long1), to_rad(lat2), to_rad(long2)
    dlat = np.abs(lat1 - lat2)
    dlong = np.abs(long1 - long2)
    return (
        2
        * EARTH_R
        * np.arcsin(
            np.sqrt(hav(dlat) + hav(dlong) * (1 - hav(dlat) - hav(lat1 + lat2)))
        )
    )


@numba.njit
def gcs_to_bearing(
    lat1: np.ndarray, long1: np.ndarray, lat2: np.ndarray, long2: np.ndarray
) -> np.ndarray:
    """Compute the bearing (forward azimuth) from one point to another, given by latitude
    and longitude in degrees.

    Formula from: https://www.movable-type.co.uk/scripts/latlong.html
    """
    lat1, long1, lat2, long2 = to_rad(lat1), to_rad(long1), to_rad(lat2), to_rad(long2)
    dlong = long2 - long1
    angle = np.arctan2(
        np.sin(dlong) * np.cos(lat2),
        np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlong),
    )
    return (to_deg(angle) + 360) % 360


@numba.njit
def apply_move(
    lat1: np.ndarray, long1: np.ndarray, bearing: np.ndarray, distance: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the position after going straight from the given point for the given
    distance, starting at the given bearing.

    Formula from: https://www.movable-type.co.uk/scripts/latlong.html
    """
    delta = distance / EARTH_R
    lat1, long1, bearing = to_rad(lat1), to_rad(long1), to_rad(bearing)
    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(delta) + np.cos(lat1) * np.sin(delta) * np.cos(bearing)
    )
    long2 = long1 + np.arctan2(
        np.sin(bearing) * np.sin(delta) * np.cos(lat1),
        np.cos(delta) - np.sin(lat1) * np.sin(lat2),
    )
    return to_deg(lat2), to_deg(long2)


@numba.njit
def gcs_to_n_vector(
    location_lat: np.ndarray, location_long: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the n-vector representation of a given latitude and longitude in degrees.

    Formula from: https://en.wikipedia.org/wiki/N-vector
    """
    location_lat, location_long = to_rad(location_lat), to_rad(location_long)
    cos_lat, cos_lon = np.cos(location_lat), np.cos(location_long)
    sin_lat, sin_lon = np.sin(location_lat), np.sin(location_long)
    return cos_lat * cos_lon, cos_lat * sin_lon, sin_lat


def n_vector_distance(location1: np.ndarray, location2: np.ndarray):
    location1, location2 = location1.astype(np.float64), location2.astype(np.float64)
    dot = np.einsum("...i,...i->...", location1, location2)
    norm = np.linalg.norm(location1, axis=-1) * np.linalg.norm(location2, axis=-1)
    nan_mask = np.isclose(norm, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.arccos(np.clip(dot / norm, 0.0, 1.0)) * EARTH_R
    result[nan_mask] = np.nan
    return result.astype(location1.dtype)


class StepSampler:
    def __init__(
        self, log_dist_quantiles: np.ndarray, turning_angle_quantiles: np.ndarray
    ):
        self.log_dist_quantiles = log_dist_quantiles
        self.turning_angle_quantiles = turning_angle_quantiles

    def sample(
        self,
        traj: Optional[Dict[str, np.ndarray]] = None,
        lat: Optional[np.ndarray] = None,
        long: Optional[np.ndarray] = None,
        bearing: Optional[np.ndarray] = None,
        num_samples: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        if rng is None:
            rng = np.random.default_rng()

        if traj:
            lat, long = traj["location_lat"], traj["location_long"]
            # To get the final bearing (arrriving at the current location), we take
            # the initial one in the opposite direction and reverse it
            bearing = gcs_to_bearing(lat[1:], long[1:], lat[:-1], long[:-1])
            with np.errstate(invalid="ignore"):
                bearing = (bearing + 180) % 360
            bearing = np.pad(bearing, [(1, 0)])

        shape, dtype = lat.shape, lat.dtype
        if num_samples is not None:
            shape = (num_samples, *shape)
            lat, long = lat[None], long[None]

        with np.errstate(invalid="ignore"):
            distance = np.exp(
                self._sample_from_quantiles(shape, self.log_dist_quantiles, rng)
            )
            turning_angle = self._sample_from_quantiles(
                shape, self.turning_angle_quantiles, rng
            )
            bearing = (bearing + turning_angle + 360) % 360
            lat, long = apply_move(
                lat.astype(np.float64), long.astype(np.float64), bearing, distance
            )
            lat, long = lat.astype(dtype), long.astype(dtype)
        if num_samples is not None:
            lat, long = lat.swapaxes(0, -1), long.swapaxes(0, -1)
        return lat, long

    @staticmethod
    def _sample_from_quantiles(shape, quantiles, rng: np.random.Generator):
        return np.interp(
            rng.random(shape), np.linspace(0, 1, len(quantiles)), quantiles
        )
