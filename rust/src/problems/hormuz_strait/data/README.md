# data

Reproducibility layout for the Strait of Hormuz problem.

- `raw/`
  - original files downloaded from external publishers
- `processed/`
  - deterministic CSV tables derived from `raw/`

The intended workflow is:

1. download raw files into `raw/`
2. build deterministic CSV views into `processed/`
3. define the modeling node set and scenario parameters from those processed tables

