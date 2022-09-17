import os
from pathlib import Path
from typing import List, Optional, Union

import gps2var

from .. import training_utils


class WorldClimBioReader(training_utils.RasterValueReaderPool):
    def __init__(
        self,
        root_path: Union[str, os.PathLike],
        var_ids: Optional[List[int]] = None,
        num_workers: int = 1,
        num_threads: Optional[int] = None,
        use_multiprocessing: bool = True,
        interpolation: str = "nearest",
    ):
        root_path = Path(root_path)
        specs = [
            {
                "path": root_path / f"wc2.1_30s_bio_{i}.tif",
                "feat_center": mean,
                "feat_scale": inv_std,
            }
            for i, mean, inv_std in zip(range(1, 20), _MEANS, _INV_STDS)
        ]
        if var_ids is not None:
            specs = [specs[i - 1] for i in var_ids]
        super().__init__(
            spec=specs,
            num_workers=num_workers,
            num_threads=num_threads,
            use_multiprocessing=use_multiprocessing,
            interpolation=interpolation,
        )


_MEANS = [
    -4.444607781058798,
    10.096086344492093,
    34.370007727122236,
    891.2090319373718,
    13.787539469205488,
    -20.40271177134371,
    34.19024379094415,
    -1.3611660708992686,
    -5.745056777226296,
    6.941681041726127,
    -14.421294752932441,
    532.131499794549,
    91.45147736774582,
    14.40531286690578,
    75.82097431192203,
    235.65372064235214,
    52.30173682400658,
    152.5487079282464,
    104.11886525145661,
]
_INV_STDS = [
    0.040476010033391005,
    0.3214225280748891,
    0.05344555694549748,
    0.0021399762797460935,
    0.04606637955532394,
    0.03843761811508031,
    0.08331273603316934,
    0.03430459261257326,
    0.04390995464841662,
    0.04717281544850558,
    0.0373186472451054,
    0.0015872099327125268,
    0.009792140739007598,
    0.036879766047647554,
    0.02270102763233621,
    0.0036459771924056417,
    0.011045701305820062,
    0.005456338375624682,
    0.005517603053323615,
]
