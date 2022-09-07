# MoveFormer

## Downloading and preparing the data

First, download the following data:
- 100d [Wikipedia2Vec embeddings](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/) in binary format ([`enwiki_20180420_100d.pkl.bz2`](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_100d.pkl.bz2)), to be placed in `data/`
- geospatial raster data, to be placed in `data/geo/`:
  - [human footprint data](https://sedac.ciesin.columbia.edu/data/set/wildareas-v3-2009-human-footprint) (`wildareas-v3-2009-human-footprint-geotiff`)
  - [WorldClim bioclimatic variables](https://www.worldclim.org/data/worldclim21.html) ([`wc2.1_30s_bio.zip`](https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_30s_bio.zip)) – 19 TIF files to be extracted inside `data/wc2.1_30s/`
  - [land cover data](https://doi.org/10.5281/zenodo.3939038) ([`PROBAV_LC100_global_v3.0.1_2015-base_Discrete-Classification-map_EPSG-4326.tif`](https://zenodo.org/record/3939038/files/PROBAV_LC100_global_v3.0.1_2015-base_Discrete-Classification-map_EPSG-4326.tif?download=1))
- TODO: describe how to get the ungulates data 

Then run the notebooks in [`data/`](./data) in the following order:
- [`movebank/download.ipynb`](./data/movebank/download.ipynb) downloads _all_ Movebank data available under a Creative Commons license. (Note that this was run on 15 Feb 2022 and will most likely produce more data today!)
- [`movebank/to_parquet.ipynb`](./data/movebank/to_parquet.ipynb) and [`ungulates/to_parquet.ipynb`](./data/ungulates/to_parquet.ipynb) read the CSV data and save it as Parquet tables.
- [`movebank/individual_vars.ipynb`](./data/movebank/individual_vars.ipynb) and [`ungulates/individual_vars.ipynb`](./data/ungulates/individual_vars.ipynb) create a mapping from individuals to Wikipedia2Vec taxon embeddings.
- [`movebank+ungulates/merge.ipynb`](./data/movebank%2Bungulates/merge.ipynb) merges the data from the two sources.
- [`movebank+ungulates/sample_12h.ipynb`](./data/movebank%2Bungulates/sample_12h.ipynb) subsamples the data with a period of approximately 12 h (noon, midnight).
- [`worldclim.ipynb`](./data/worldclim.ipynb) processes the WorldClim bioclimatic variables.
