# MoveFormer

## Preparing the data

First, download 100d Wikipedia2Vec embeddings in binary format (`enwiki_20180420_100d.pkl.bz2`) [here](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/) and place them in `data/`.

TODO: describe how to get the ungulates data 

Then run the notebooks in [`data/`](./data) in the following order:
- [`movebank/download.ipynb`](./data/movebank/download.ipynb) downloads _all_ Movebank data available under a Creative Commons license. (Note that this was run on 15 Feb 2022 and will most likely produce more data today!)
- [`movebank/to_parquet.ipynb`](./data/movebank/to_parquet.ipynb) and [`ungulates/to_parquet.ipynb`](./data/ungulates/to_parquet.ipynb) read the CSV data and save it as Parquet tables.
- [`movebank/individual_vars.ipynb`](./data/movebank/individual_vars.ipynb) and [`ungulates/individual_vars.ipynb`](./data/ungulates/individual_vars.ipynb) create a mapping from individuals to Wikipedia2Vec taxon embeddings.
- [`movebank+ungulates/merge.ipynb`](./data/movebank%2Bungulates/merge.ipynb) merges the data from the two sources.
- [`movebank+ungulates/sample_12h.ipynb`](./data/movebank%2Bungulates/sample_12h.ipynb) subsamples the data with a period of approximately 12 h (noon, midnight).
