# Supported analysis pipeline

This repository now has one supported route from raw parliamentary text to the
inconsistency results. Historical output files are preserved, but legacy code
paths are not.

## 1. Data preparation

- `scripts/api_handling/client_mint.py` extracts ParlaMint speeches.
- `scripts/api_handling/party_mapping_builder.py` maps speech-side party labels
  to Manifesto Project party identifiers.
- `scripts/api_handling/manifesto_quasi_builder.py` downloads/processes
  manifesto quasi-sentences and creates the strict speech-date bridge.

Run the mapping and manifesto stages with:

```bash
make data COUNTRY=CZ
```

## 2. PLDA

- `scripts/ml_algos/plda_test.py` trains the supervised PLDA model.
- `scripts/ml_algos/plda_inference.py` infers manifesto topic distributions.
- The retained `scripts/metrics/plda_*` modules build the topic, alignment, and
  regression-panel inputs used downstream.

```bash
make plda COUNTRY=CZ
```

## 3. Strict candidate retrieval

`scripts/inconsistency/poc_nli.py` links each speech to the latest
MPDS-election-dated manifesto on or before the speech date and retrieves
manifesto candidates. The supported Make target always passes
`--retrieval-only`: generic NLI does not supply the final population label.

```bash
make retrieve COUNTRY=CZ TOPIC=Macroeconomics
```

Retrieval is checkpointed and bound to its inputs/configuration. A mismatched
checkpoint fails instead of being silently reused.

## 4. Inconsistency classifier

- `scripts/inconsistency/llm_consensus_labeling.py` preserves the completed
  three-provider silver-label workflow.
- `scripts/inconsistency/train_llm_consensus_deberta.py` contains the canonical
  manifesto-document-grouped split/training implementation.
- `scripts/inconsistency/train_manifesto_grouped_deberta.py` is the retained
  entry point for the existing corrected model artifacts.
- `scripts/inconsistency/classify_llm_consensus_deberta.py` classifies the full
  retrieved population with model-bound, resumable checkpoints.

All classifier splits keep a complete manifesto document in exactly one split
and fail on leakage. A complete saved model/test result is treated as complete,
so the training commands do not restart it.

```bash
make classifier_prepare
make classifier_train
make classify_country COUNTRY=CZ TOPIC=Macroeconomics
```

The “older classifier” means the generic off-the-shelf NLI labels and the
unverified historical `nli_consensus_classifier` output root. They are retained
only as output artifacts; the supported primary root is
`nli_consensus_classifier_manifesto_grouped`.

## 5. Validation and final audits

- `prepare_macro_human_validation.py` and
  `analyze_macro_human_validation.py` implement blind human validation.
- `methodology_freeze_robustness.py` enforces provenance, temporal-linkage,
  leakage, and population-reproduction assertions.
- `empirical_regression_audit.py` is the retained consolidated empirical
  analysis.

```bash
make human_validation_prepare
make human_validation_analyze
make methodology_audit
make empirical_audit
make test
```

Use `make help` for the short command list.
