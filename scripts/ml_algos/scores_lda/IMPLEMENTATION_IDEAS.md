# LDA Evaluation Implementation Ideas

This note maps the planned LDA metrics onto the current codebase, where the main evaluation script is [`ml_algos/lda_party_month_analysis.py`](/Users/anna/projs/BA/BaThesis/scripts/ml_algos/lda_party_month_analysis.py) and the analysis unit is the aggregated party-month document produced by `aggregate_party_month()`.

## Current Unit Of Analysis

The script does not model individual speeches. It concatenates speeches into one document per `party x month`. That choice affects every metric:

- held-out perplexity becomes predictive performance on held-out party-month documents
- NPMI coherence is computed from word co-occurrence across party-month documents
- topic purity is purity over party labels assigned to aggregated documents
- fragmentation entropy measures how diffuse each party-month document is across topics

If that is the intended analysis level, the metrics below are coherent. If not, the aggregation step should change before adding more metrics.

## Recommended File Layout

Keep [`ml_algos/lda_party_month_analysis.py`](/Users/anna/projs/BA/BaThesis/scripts/ml_algos/lda_party_month_analysis.py) as orchestration only:

- load and aggregate data
- choose the profile to score: all parties, top 1, top N
- iterate over topic counts
- save plots and CSV outputs

Put metric logic in [`ml_algos/scores_lda/`](/Users/anna/projs/BA/BaThesis/scripts/ml_algos/scores_lda):

- `held_out_perplexity.py`
- `intrinsic_metrics.py`
- `stability.py`
- optional later: `runner.py`

## Shared Run Artifact

The clean implementation pattern is to compute all metrics from the same per-run artifact.

Suggested dataclass:

```python
@dataclass(frozen=True)
class LdaRunArtifacts:
    seed: int
    n_topics: int
    vectorizer: CountVectorizer
    lda: LatentDirichletAllocation
    train_df: pd.DataFrame
    test_df: pd.DataFrame | None
    X_train: spmatrix
    X_test: spmatrix | None
    doc_topic_train: np.ndarray
    doc_topic_test: np.ndarray | None
    topic_word: np.ndarray
    feature_names: np.ndarray
    top_words_per_topic: list[list[str]]
```

That gives one consistent basis for every metric instead of fitting separate models inside each metric.

## Metric Definitions

### 1. Held-Out Perplexity

- split aggregated documents into train and test for each seed
- fit `CountVectorizer` on train only
- fit `LatentDirichletAllocation` on `X_train`
- transform `X_test` with the train vocabulary
- drop held-out docs with zero vocabulary overlap
- score `lda.perplexity(X_test_nonempty)`
- aggregate mean and SD across seeds

This metric is a sensible primary selector for topic count in the current script.

### 2. Average NPMI Coherence

- extract the top `k` words per topic from `lda.components_`
- compute pairwise NPMI for all top-word pairs using document co-occurrence on the training corpus
- coherence of one topic = mean pairwise NPMI
- run coherence = mean coherence over topics
- final score = mean and SD across seeds

This should be computed on the training corpus of each run, not on held-out documents.

### 3. Average Topic Purity

Recommended definition in this codebase: purity with respect to `party`.

- assign each aggregated document its dominant topic via `argmax(doc_topic)`
- for each topic, count the party labels of documents assigned to it
- topic purity = `max_party_count / total_docs_assigned_to_topic`
- run purity = mean over topics
- final score = mean and SD across seeds

If you want temporal purity instead, the same logic can be applied to `month`.

### 4. Average Fragmentation Entropy

Use normalized entropy of each document's topic distribution:

- for each aggregated document, let `theta` be its topic mixture
- document entropy = `-sum(theta_k * log(theta_k)) / log(K)`
- run fragmentation entropy = mean document entropy
- final score = mean and SD across seeds

Interpretation:

- low entropy means concentrated documents
- high entropy means fragmented documents

### 5. Average Stability

Recommended definition: topic-set stability across seeds after topic matching.

- for the same `n_topics`, fit one model per seed
- extract top `k` words per topic
- build a topic similarity matrix between two runs using Jaccard overlap or rank-biased overlap
- use Hungarian matching for one-to-one topic alignment
- pairwise run stability = mean similarity of matched topic pairs
- final stability = mean and SD over all run pairs

This is the least misleading stability definition for multi-run LDA.

### 6. Topic Diversity

Use the standard unique top-word proportion:

- collect the top `k` words from every topic in one run
- `topic_diversity = unique_top_words / (k * n_topics)`
- final score = mean and SD across seeds

This complements coherence: coherence rewards internally tight topics, diversity penalizes redundant topics.

## Output Schema

Prefer one tidy CSV for all evaluation metrics:

```text
profile,n_topics,metric,mean,sd,runs
all_parties,45,held_out_perplexity,1234.56,31.22,10
all_parties,45,npmi_coherence,0.081,0.012,10
...
```

That is easier to plot and easier to compare than nested dict dumps.

## Selection Strategy

Do not combine all metrics into one selector immediately.

Recommended order:

- use held-out perplexity as the primary selector
- inspect NPMI, diversity, purity, fragmentation entropy, and stability as diagnostics
- only later introduce a composite ranking if you have a principled weighting scheme

## Immediate Refactor

The first refactor is modest:

- move perplexity code out of [`ml_algos/lda_party_month_analysis.py`](/Users/anna/projs/BA/BaThesis/scripts/ml_algos/lda_party_month_analysis.py) into [`ml_algos/scores_lda/held_out_perplexity.py`](/Users/anna/projs/BA/BaThesis/scripts/ml_algos/scores_lda/held_out_perplexity.py)
- keep the script responsible for profile construction and plotting
- have the score module return a typed summary per topic count

That makes the later metric additions much easier because the orchestration layer no longer owns metric internals.
