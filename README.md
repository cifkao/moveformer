# MoveFormer

This repository accompanies the paper [*MoveFormer: a Transformer-based model for step-selection animal movement modelling*](https://www.biorxiv.org/content/10.1101/2023.03.05.531080v1) by Ondřej Cífka, Simon Chamaillé-Jammes, and Antoine Liutkus.

The repository contains the following:
- `geo_transformers` – the main Python package
- `config` – training configuration files
- `exp` – where models live
- `data` – data preparation Jupyter notebooks
- `sandbox` – other Jupyter notebooks (mainly to compute results)

See also [`gps2var`](https://github.com/cifkao/gps2var), a package for fast loading of geospatial variables by GPS coordinates, also developed as part of this project.

## Installation

First, install [GDAL](https://gdal.org/) according to the [instructions](https://gdal.org/download.html#binaries) appropriate for your system (e.g. `apt install gdal-bin libgdal-dev`, `conda install -c conda-forge gdal`, `brew install gdal`, ...). Then proceed with **either Pip or Poetry** as follows.

### Pip
Make sure you have Python 3.9, `pip>=21.2`, and `setuptools==59.5.0` (run `pip install -U pip 'setuptools==59.5.0'`). Then:
- To install only the dependencies required for training, run:
  ```bash
  pip install -r requirements.txt
  pip install -e .
  ```
- To install all dependencies (including those required to run data preparation and evaluation notebooks):
  ```bash
  pip install -r requirements-notebook.txt
  pip install -e '.[notebook]'
  ```

### Poetry
Install [Poetry](https://python-poetry.org/) 1.1.13 or later.

Optionally (if you are not in a virtual environment or your default Python version is different from 3.9) set up a virtual environment and activate it:
```bash
poetry env use python3.9
poetry shell
```
Then run:
```bash
pip install -U pip 'setuptools==59.5.0'
```
Finally, run either:
- `poetry install` to install only the dependencies required for training.
- `poetry install -E notebook` to install all dependencies (including those required to run data preparation and evaluation notebooks).

## Downloading and preparing the data

The following data is required for both training and inference:
- 100d [Wikipedia2Vec embeddings](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/) in binary format ([`enwiki_20180420_100d.pkl.bz2`](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_100d.pkl.bz2)), to be placed in `data/`
- geospatial raster data, to be placed in `data/geo/`:
  - [human footprint data](https://sedac.ciesin.columbia.edu/data/set/wildareas-v3-2009-human-footprint) (`wildareas-v3-2009-human-footprint-geotiff`)
  - [WorldClim bioclimatic variables](https://www.worldclim.org/data/worldclim21.html) ([`wc2.1_30s_bio.zip`](https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_30s_bio.zip)) – 19 TIF files to be extracted inside `data/geo/wc2.1_30s/`
  - [land cover data](https://doi.org/10.5281/zenodo.3939038) ([`PROBAV_LC100_global_v3.0.1_2015-base_Discrete-Classification-map_EPSG-4326.tif`](https://zenodo.org/record/3939038/files/PROBAV_LC100_global_v3.0.1_2015-base_Discrete-Classification-map_EPSG-4326.tif?download=1))

The notebooks in [`data/`](./data) show how the training data was acquired and pre-processed. The notebooks in [`ungulates/`](./data/ungulates) can be used as an example of how to pre-process custom (non-Movebank) data.
- [`movebank/download.ipynb`](./data/movebank/download.ipynb) downloads _all_ Movebank data available under a Creative Commons license. (Note that this was run on 15 Feb 2022 and will most likely produce more data today!)
- [`movebank/to_parquet.ipynb`](./data/movebank/to_parquet.ipynb) and [`ungulates/to_parquet.ipynb`](./data/ungulates/to_parquet.ipynb) read the CSV data and save it as Parquet tables.
- [`movebank/individual_vars.ipynb`](./data/movebank/individual_vars.ipynb) and [`ungulates/individual_vars.ipynb`](./data/ungulates/individual_vars.ipynb) create a mapping from individuals to Wikipedia2Vec taxon embeddings.
- [`movebank+ungulates/merge.ipynb`](./data/movebank%2Bungulates/merge.ipynb) merges the data from the two sources.
- [`movebank+ungulates/sample_12h.ipynb`](./data/movebank%2Bungulates/sample_12h.ipynb) subsamples the data with a period of approximately 12 h (noon, midnight).
- [`geo/worldclim.ipynb`](./data/geo/worldclim.ipynb) processes the WorldClim bioclimatic variables.

## Pretrained models

We include all models used in our paper, each living in a directory under `exp/`:
- VarCtx: `37ld98g9`
- FullCtx: `vdv96ee1`
- NoAtt: `2u6ajp2u`
- NoEnc: `2rygu3l4`

To get the model weights, download the `checkpoints.zip` archive [from Zenodo](https://doi.org/10.5281/zenodo.7698263) and extract it in the root of the repository. On Linux:
```bash
wget https://zenodo.org/record/7698263/files/checkpoints.zip
unzip checkpoints.zip
```

## Training a model

Before training a model, either set up [Weights and Biases](https://wandb.ai/) or disable it using `export WANDB_MODE=disabled`. Then run the `python -m geo_transformers.models.any_horizon_forecast_transformer fit` command with an appropriate configuration file.

The commands to train the models from the paper are:
```bash
# VarCtx (37ld98g9)
python -m geo_transformers.models.any_horizon_forecast_transformer fit --config config/forecast_mbk+ung_12h_sel_nofut.yaml --trainer.gpus=1 --trainer.logger=WandbLogger --trainer.logger.project=geo-transformers --trainer.logger.save_dir=exp/forecast_mbk+ung_12h
```
```bash
# FullCtx (vdv96ee1)
python -m geo_transformers.models.any_horizon_forecast_transformer fit --config config/forecast_mbk+ung_12h_sel_nofut.yaml --trainer.gpus=1 --trainer.logger=WandbLogger --trainer.logger.project=geo-transformers --trainer.logger.save_dir=exp/forecast_mbk+ung_12h --model.var_len_training_v2=False
```
```bash
# NoAtt (2u6ajp2u)
python -m geo_transformers.models.any_horizon_forecast_transformer fit --config config/forecast_mbk+ung_12h_sel_nofut_ctx1.yaml --trainer.gpus=1 --trainer.logger=WandbLogger --trainer.logger.project=geo-transformers --trainer.logger.save_dir=exp/forecast_mbk+ung_12h
```
```bash
# NoEnc (2rygu3l4)
python -m geo_transformers.models.any_horizon_forecast_transformer fit --config config/forecast_mbk+ung_12h_sel_nofut_ctx1.yaml --trainer.gpus=1 --trainer.logger=WandbLogger --trainer.logger.project=geo-transformers --trainer.logger.save_dir=exp/forecast_mbk+ung_12h --model.encoder.depth=0
```

### Candidate samplers

[`sandbox/movebank+ungulates_12h_id2sampler.pickle`](./sandbox/movebank+ungulates_12h_id2sampler.pickle) contains a mapping from sequence IDs to `StepSampler` instances, which serve to sample candidate locations.

To (re)estimate the samplers, run the [`sandbox/movebank+ungulates_12h_stats.ipynb`](./sandbox/movebank+ungulates_12h_stats.ipynb) notebook. It estimates the distributions of turning angles and log-distances for each taxon in the data, creates the corresponding `StepSampler` objects, and saves them by sequence ID.

## Results

The code for running the models on the test data and computing results is contained in notebooks in the [`sandbox`](./sandbox) directory:
- [`mbk+ung_12h_eval.ipynb`](./sandbox/mbk+ung_12h_eval.ipynb) – general validation
- [`mbk+ung_12h_cent.ipynb`](./sandbox/mbk+ung_12h_cent.ipynb) – context length analysis
- [`mbk+ung_12h_feat_imp.ipynb`](./sandbox/mbk+ung_12h_feat_imp.ipynb) – candidate feature importance

Pre-computed metrics are provided so that the analyses can be replicated without running the models.

## Acknowledgments
This work was supported by the LabEx NUMEV (ANR-10-LABX-0020) and the REPOS project, both funded by the I-Site MUSE (ANR-16-IDEX-0006). Computations were performed using HPC/AI resources from GENCI-IDRIS (Grant AD011012019R1).

We would like to thank all authors who made their data available through Movebank under Creative Commons licenses.
