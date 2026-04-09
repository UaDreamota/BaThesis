from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    std: float
    runs: int


@dataclass(frozen=True)
class CoherenceSummary:
    npmi: MetricSummary
    c_v: MetricSummary


def build_vectorizer(stop_words: list[str] | None) -> CountVectorizer:
    return CountVectorizer(stop_words=stop_words or None)


def build_lda_model(
    n_topics: int,
    random_seed: int,
    doc_topic_prior: float | None = None,
    topic_word_prior: float | None = None,
) -> LatentDirichletAllocation:
    return LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_seed,
        learning_method="batch",
        doc_topic_prior=doc_topic_prior,
        topic_word_prior=topic_word_prior,
    )


def summarize_metric(values: list[float]) -> MetricSummary:
    return MetricSummary(
        mean=float(np.mean(values)),
        std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        runs=len(values),
    )


def prepare_texts(texts: list[str] | tuple[str, ...] | np.ndarray) -> list[str]:
    prepared = [str(text).strip() for text in texts if str(text).strip()]
    if not prepared:
        raise ValueError("Coherence evaluation requires at least one non-empty document.")
    return prepared


def extract_top_word_indices(
    lda: LatentDirichletAllocation,
    top_n_words: int,
) -> list[list[int]]:
    return [
        component.argsort()[-top_n_words:][::-1].tolist()
        for component in lda.components_
    ]


def compute_npmi(
    doc_frequency_i: float,
    doc_frequency_j: float,
    pair_frequency: float,
    n_documents: int,
) -> float:
    if n_documents <= 0 or doc_frequency_i <= 0 or doc_frequency_j <= 0:
        return 0.0
    if pair_frequency <= 0:
        return -1.0

    p_i = doc_frequency_i / n_documents
    p_j = doc_frequency_j / n_documents
    p_ij = pair_frequency / n_documents
    pmi = np.log(p_ij / (p_i * p_j))
    return float(pmi / -np.log(p_ij))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def score_topic_npmi(
    topic_indices: list[int],
    doc_frequencies: np.ndarray,
    pair_frequencies: np.ndarray,
    n_documents: int,
) -> float:
    if len(topic_indices) < 2:
        return 0.0

    pair_scores = [
        compute_npmi(
            doc_frequency_i=doc_frequencies[i],
            doc_frequency_j=doc_frequencies[j],
            pair_frequency=pair_frequencies[i, j],
            n_documents=n_documents,
        )
        for i, j in combinations(topic_indices, 2)
    ]
    return float(np.mean(pair_scores)) if pair_scores else 0.0


def score_topic_c_v(
    topic_indices: list[int],
    doc_frequencies: np.ndarray,
    pair_frequencies: np.ndarray,
    n_documents: int,
) -> float:
    if len(topic_indices) < 2:
        return 0.0

    # This is a self-contained c_v-style score: word context vectors are built
    # from document-level NPMI against the other top words in the topic.
    context_vectors: list[np.ndarray] = []
    for word_index in topic_indices:
        vector = np.array(
            [
                1.0
                if other_index == word_index
                else max(
                    compute_npmi(
                        doc_frequency_i=doc_frequencies[word_index],
                        doc_frequency_j=doc_frequencies[other_index],
                        pair_frequency=pair_frequencies[word_index, other_index],
                        n_documents=n_documents,
                    ),
                    0.0,
                )
                for other_index in topic_indices
            ],
            dtype=float,
        )
        context_vectors.append(vector)

    pair_scores = [
        cosine_similarity(context_vectors[left_idx], context_vectors[right_idx])
        for left_idx, right_idx in combinations(range(len(context_vectors)), 2)
    ]
    return float(np.mean(pair_scores)) if pair_scores else 0.0


def score_run_coherence(
    X,
    n_topics: int,
    random_seed: int,
    top_n_words: int,
    doc_topic_prior: float | None = None,
    topic_word_prior: float | None = None,
) -> tuple[float, float]:
    lda = build_lda_model(
        n_topics,
        random_seed,
        doc_topic_prior=doc_topic_prior,
        topic_word_prior=topic_word_prior,
    )
    lda.fit(X)

    top_word_indices = extract_top_word_indices(lda, top_n_words=top_n_words)
    unique_word_indices = sorted({index for topic in top_word_indices for index in topic})
    if not unique_word_indices:
        return 0.0, 0.0

    X_selected = X[:, unique_word_indices]
    presence = X_selected.copy()
    presence.data = np.ones_like(presence.data)

    doc_frequencies = np.asarray(presence.sum(axis=0)).ravel().astype(float)
    pair_frequencies = (presence.T @ presence).toarray().astype(float)
    n_documents = presence.shape[0]
    global_to_local = {global_index: local_index for local_index, global_index in enumerate(unique_word_indices)}

    topic_npmis: list[float] = []
    topic_c_vs: list[float] = []
    for topic in top_word_indices:
        topic_local_indices = [global_to_local[index] for index in topic]
        topic_npmis.append(
            score_topic_npmi(
                topic_indices=topic_local_indices,
                doc_frequencies=doc_frequencies,
                pair_frequencies=pair_frequencies,
                n_documents=n_documents,
            )
        )
        topic_c_vs.append(
            score_topic_c_v(
                topic_indices=topic_local_indices,
                doc_frequencies=doc_frequencies,
                pair_frequencies=pair_frequencies,
                n_documents=n_documents,
            )
        )

    return float(np.mean(topic_npmis)), float(np.mean(topic_c_vs))


def compute_coherence_profile(
    texts: list[str] | tuple[str, ...] | np.ndarray,
    stop_words: list[str] | None,
    topic_counts: list[int],
    n_runs: int,
    base_random_seed: int = 42,
    top_n_words: int = 10,
    doc_topic_prior: float | None = None,
    topic_word_prior: float | None = None,
) -> dict[int, CoherenceSummary]:
    prepared_texts = prepare_texts(texts)
    if n_runs <= 0:
        raise ValueError("Coherence evaluation requires n_runs > 0.")
    if top_n_words < 2:
        raise ValueError("Coherence evaluation requires top_n_words >= 2.")

    vectorizer = build_vectorizer(stop_words)
    X = vectorizer.fit_transform(prepared_texts)

    npmi_scores_by_topic: dict[int, list[float]] = {n_topics: [] for n_topics in topic_counts}
    c_v_scores_by_topic: dict[int, list[float]] = {n_topics: [] for n_topics in topic_counts}

    for run_idx in range(n_runs):
        run_seed = base_random_seed + run_idx
        for n_topics in topic_counts:
            run_npmi, run_c_v = score_run_coherence(
                X=X,
                n_topics=n_topics,
                random_seed=run_seed,
                top_n_words=top_n_words,
                doc_topic_prior=doc_topic_prior,
                topic_word_prior=topic_word_prior,
            )
            npmi_scores_by_topic[n_topics].append(run_npmi)
            c_v_scores_by_topic[n_topics].append(run_c_v)

    return {
        n_topics: CoherenceSummary(
            npmi=summarize_metric(npmi_scores_by_topic[n_topics]),
            c_v=summarize_metric(c_v_scores_by_topic[n_topics]),
        )
        for n_topics in topic_counts
    }
