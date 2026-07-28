# Thesis reproducibility scripts

This package contains the code used for the thesis analysis while preserving the original project structure. The research data, trained model artifacts, caches, and generated outputs are intentionally excluded because the full data volume exceeds 200 GB.

## Included

- `scripts/`: data preparation, PLDA, retrieval, inconsistency classification, validation, regression, robustness, and visualization code
- `tests/`: unit tests for key validation and estimation logic
- `substantive graphs/build_substantive_graphs.py`: thesis figure-generation script
- `urs_orchestrator.py`: orchestration entry point
- `Makefile` and `PIPELINE.md`: supported execution order and commands
- `requirements.txt`: Python dependencies
- `scripts/party_mappings/` and `scripts/stopwords/`: small code-side resources required by the pipeline

## Not included

- Parliamentary speech corpora and Manifesto Project source data
- Intermediate and final analysis datasets
- Embedding caches, checkpoints, trained PLDA/DeBERTa models, figures, PDFs, and other generated outputs
- `.env` and all API credentials
- The local Python virtual environment

## Setup

1. Create a Python environment and install `requirements.txt`.
2. Copy `.env.example` to `.env` and provide only the variables needed for the stages being run.
3. Place or mount the data separately and set `DATA_FOLDER` and/or `PARLAM_DATA_PATH`.
4. Read `PIPELINE.md`; run `make help` for the supported pipeline targets.

The empty `data/` and `outputs/` directories indicate the paths expected by the default configuration. Large inputs must be obtained separately from their original providers or the thesis author.