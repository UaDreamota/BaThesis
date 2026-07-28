PYTHON ?= .venv/bin/python
COUNTRY ?= CZ
TOPIC ?= Macroeconomics
CLUSTER ?= all

PAIR_ROOT ?= outputs/test_speeches/nli_inconsistency
CLASSIFIER_TRAIN_DIR ?= outputs/test_speeches/methodology_freeze_robustness/manifesto_grouped_deberta
CLASSIFIER_MODEL ?= $(CLASSIFIER_TRAIN_DIR)/model
CLASSIFIER_OUTPUT_ROOT ?= outputs/test_speeches/nli_consensus_classifier_manifesto_grouped
CONSENSUS_SAMPLE ?= outputs/test_speeches/nli_inconsistency/llm_consensus/llm_contradiction80_emb040_final_sample.csv
CONSENSUS_PROVIDER_DIR ?= outputs/test_speeches/nli_inconsistency/llm_consensus/provider_labels_contradiction80_emb040_final
CONSENSUS_LABELS ?= outputs/test_speeches/nli_inconsistency/llm_consensus/llm_contradiction80_emb040_consensus_labels.csv

NLI_MODEL ?= MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
EMBEDDING_PROVIDER ?= openai
EMBEDDING_MODEL ?= text-embedding-3-small
DEVICE ?= auto
RETRIEVAL_TOP_K ?= 3
CLASSIFIER_BATCH_SIZE ?= 8
CLASSIFIER_CHUNK_SIZE ?= 2000

.PHONY: help data party_mapping manifesto plda plda_train plda_infer \
	plda_metrics retrieve retrieve_macro retrieve_galtan consensus_sample \
	consensus_label consensus_merge classifier_prepare classifier_train \
	classify_country classify_cluster human_validation_prepare \
	human_validation_analyze methodology_audit plda_salience_panels \
	empirical_audit plda_headline_verify test

help:
	@echo "Supported pipeline:"
	@echo "  make data COUNTRY=CZ"
	@echo "  make plda COUNTRY=CZ"
	@echo "  make retrieve COUNTRY=CZ TOPIC=Macroeconomics"
	@echo "  make classify_country COUNTRY=CZ TOPIC=Macroeconomics"
	@echo "  make methodology_audit"
	@echo ""
	@echo "Retrieval is strict latest-preceding-manifesto and retrieval-only."
	@echo "The final inconsistency labels come from CLASSIFIER_MODEL."

data:
	$(MAKE) party_mapping COUNTRY=$(COUNTRY)
	$(MAKE) manifesto COUNTRY=$(COUNTRY)

party_mapping:
	$(PYTHON) scripts/api_handling/party_mapping_builder.py --country $(COUNTRY)

manifesto:
	$(PYTHON) scripts/api_handling/manifesto_quasi_builder.py --country $(COUNTRY)

plda:
	$(MAKE) plda_train COUNTRY=$(COUNTRY)
	$(MAKE) plda_infer COUNTRY=$(COUNTRY)
	$(MAKE) plda_metrics COUNTRY=$(COUNTRY)

plda_train:
	$(PYTHON) scripts/ml_algos/plda_test.py --c $(COUNTRY) \
		--n-latent-topics 1 --topics-per-label 2 --rm-top 2 \
		--alpha 0.001 --eta 0.01

plda_infer:
	$(PYTHON) scripts/ml_algos/plda_inference.py --c $(COUNTRY)

plda_metrics:
	$(PYTHON) scripts/vizualization/plda_distribution.py --c $(COUNTRY)
	$(PYTHON) scripts/metrics/plda_weighted_speech_topics.py \
		--country $(COUNTRY) --weightings log_word_count
	$(PYTHON) scripts/metrics/plda_manifesto_alignment.py --c $(COUNTRY)
	$(PYTHON) scripts/metrics/plda_regression_panel.py --country $(COUNTRY)

# This is the population candidate-retrieval stage. The strict speech-date
# bridge is enforced inside poc_nli.py. Generic NLI inference is deliberately
# skipped; the manifesto-grouped classifier supplies the final labels.
retrieve:
	$(PYTHON) scripts/inconsistency/poc_nli.py \
		--country $(COUNTRY) --topic $(TOPIC) \
		--speech-unit segment --sample-size 0 \
		--pair-selection hybrid_embedding_bm25 \
		--embedding-provider $(EMBEDDING_PROVIDER) \
		--embedding-model $(EMBEDDING_MODEL) \
		--device $(DEVICE) --hybrid-rerank-top-k $(RETRIEVAL_TOP_K) \
		--model-name $(NLI_MODEL) --output-dir $(PAIR_ROOT) \
		--retrieval-only

retrieve_macro:
	$(MAKE) retrieve COUNTRY=$(COUNTRY) TOPIC=Macroeconomics

retrieve_galtan:
	$(MAKE) retrieve COUNTRY=$(COUNTRY) TOPIC=Gal_Tan

consensus_sample:
	$(PYTHON) scripts/inconsistency/llm_consensus_labeling.py sample \
		--target-size 8000 --max-scan-rows-per-file 75000 \
		--label-shares contradiction=0.8,entailment=0.1,neutral=0.1 \
		--max-retrieval-rank 2 --min-embedding-score 0.40 \
		--output $(CONSENSUS_SAMPLE)

consensus_label:
	$(PYTHON) scripts/inconsistency/llm_consensus_labeling.py label \
		--input $(CONSENSUS_SAMPLE) --output-dir $(CONSENSUS_PROVIDER_DIR) \
		--providers all --workers 1

consensus_merge:
	$(PYTHON) scripts/inconsistency/llm_consensus_labeling.py consensus \
		--input $(CONSENSUS_SAMPLE) --output-dir $(CONSENSUS_PROVIDER_DIR) \
		--output $(CONSENSUS_LABELS)

# This entry point preserves and resumes the existing manifesto-grouped
# training checkpoints. It fails if the split or hyperparameters changed.
classifier_prepare:
	$(PYTHON) scripts/inconsistency/train_manifesto_grouped_deberta.py \
		--output-dir $(CLASSIFIER_TRAIN_DIR) --prepare-only

classifier_train:
	$(PYTHON) scripts/inconsistency/train_manifesto_grouped_deberta.py \
		--output-dir $(CLASSIFIER_TRAIN_DIR)

classify_country:
	$(PYTHON) scripts/inconsistency/classify_llm_consensus_deberta.py \
		--countries $(COUNTRY) --topic $(TOPIC) \
		--model $(CLASSIFIER_MODEL) --input-root $(PAIR_ROOT) \
		--output-root $(CLASSIFIER_OUTPUT_ROOT) --device $(DEVICE) \
		--batch-size $(CLASSIFIER_BATCH_SIZE) \
		--chunk-size $(CLASSIFIER_CHUNK_SIZE)

classify_cluster:
	$(PYTHON) scripts/inconsistency/classify_llm_consensus_deberta.py \
		--cluster $(CLUSTER) --topic $(TOPIC) \
		--model $(CLASSIFIER_MODEL) --input-root $(PAIR_ROOT) \
		--output-root $(CLASSIFIER_OUTPUT_ROOT) --device $(DEVICE) \
		--batch-size $(CLASSIFIER_BATCH_SIZE) \
		--chunk-size $(CLASSIFIER_CHUNK_SIZE)

human_validation_prepare:
	$(PYTHON) scripts/inconsistency/prepare_macro_human_validation.py

human_validation_analyze:
	$(PYTHON) scripts/inconsistency/analyze_macro_human_validation.py

methodology_audit:
	$(PYTHON) scripts/inconsistency/methodology_freeze_robustness.py \
		--pair-root $(CLASSIFIER_OUTPUT_ROOT) \
		--model $(CLASSIFIER_MODEL) --split-dir $(CLASSIFIER_TRAIN_DIR)

plda_salience_panels:
	$(PYTHON) scripts/metrics/plda_salience_panels.py

empirical_audit: plda_salience_panels
	$(PYTHON) scripts/causality/empirical_regression_audit.py \
		--output-dir outputs/test_speeches/empirical_regression_audit \
		--bootstrap-reps 999 --seed 1711

plda_headline_verify: empirical_audit
	$(PYTHON) scripts/metrics/verify_plda_headline.py

test:
	$(PYTHON) -m unittest discover -s tests -v
