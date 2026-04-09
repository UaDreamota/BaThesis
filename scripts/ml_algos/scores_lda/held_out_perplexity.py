from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class PerplexitySummary:
    mean: float
    std: float
    runs: int


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


def summarize_perplexities(scores: list[float]) -> PerplexitySummary:
    return PerplexitySummary(
        mean=float(np.mean(scores)),
        std=float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        runs=len(scores),
    )


def prepare_texts(texts: list[str] | tuple[str, ...] | np.ndarray) -> list[str]:
    prepared = [str(text).strip() for text in texts if str(text).strip()]
    if not prepared:
        raise ValueError("Perplexity evaluation requires at least one non-empty document.")
    return prepared


def compute_in_sample_perplexity(
    texts: list[str] | tuple[str, ...] | np.ndarray,
    stop_words: list[str] | None,
    topic_counts: list[int],
    random_seed: int = 42,
    doc_topic_prior: float | None = None,
    topic_word_prior: float | None = None,
) -> dict[int, PerplexitySummary]:
    prepared_texts = prepare_texts(texts)
    vectorizer = build_vectorizer(stop_words)
    X = vectorizer.fit_transform(prepared_texts)

    results: dict[int, PerplexitySummary] = {}
    for n_topics in topic_counts:
        lda = build_lda_model(
            n_topics,
            random_seed,
            doc_topic_prior=doc_topic_prior,
            topic_word_prior=topic_word_prior,
        )
        lda.fit(X)
        results[n_topics] = summarize_perplexities([float(lda.perplexity(X))])
    return results


def compute_held_out_perplexity(
    texts: list[str] | tuple[str, ...] | np.ndarray,
    stop_words: list[str] | None,
    topic_counts: list[int],
    n_runs: int,
    base_random_seed: int = 42,
    test_size: float = 0.2,
    doc_topic_prior: float | None = None,
    topic_word_prior: float | None = None,
) -> dict[int, PerplexitySummary]:
    prepared_texts = prepare_texts(texts)

    if n_runs <= 0:
        raise ValueError("Held-out perplexity requires n_runs > 0.")
    if len(prepared_texts) < 2:
        raise ValueError("Held-out perplexity requires at least two documents.")

    scores_by_topic: dict[int, list[float]] = {n_topics: [] for n_topics in topic_counts}

    for run_idx in range(n_runs):
        run_seed = base_random_seed + run_idx
        train_texts, test_texts = train_test_split(
            prepared_texts,
            test_size=test_size,
            random_state=run_seed,
            shuffle=True,
        )

        vectorizer = build_vectorizer(stop_words)
        X_train = vectorizer.fit_transform(train_texts)
        X_test = vectorizer.transform(test_texts)

        if X_train.shape[1] == 0:
            raise ValueError("Held-out perplexity could not build a training vocabulary.")

        test_token_mask = np.asarray(X_test.sum(axis=1)).ravel() > 0
        X_test_eval = X_test[test_token_mask]
        if X_test_eval.shape[0] == 0:
            raise ValueError(
                "Held-out perplexity split produced no test documents with vocabulary overlap."
            )

        for n_topics in topic_counts:
            lda = build_lda_model(
                n_topics,
                run_seed,
                doc_topic_prior=doc_topic_prior,
                topic_word_prior=topic_word_prior,
            )
            lda.fit(X_train)
            scores_by_topic[n_topics].append(float(lda.perplexity(X_test_eval)))

    return {
        n_topics: summarize_perplexities(scores)
        for n_topics, scores in scores_by_topic.items()
    }


def compute_perplexity_profile(
    texts: list[str] | tuple[str, ...] | np.ndarray,
    stop_words: list[str] | None,
    topic_counts: list[int],
    held_out_runs: int = 0,
    base_random_seed: int = 42,
    test_size: float = 0.2,
    doc_topic_prior: float | None = None,
    topic_word_prior: float | None = None,
) -> dict[int, PerplexitySummary]:
    if held_out_runs > 0:
        return compute_held_out_perplexity(
            texts=texts,
            stop_words=stop_words,
            topic_counts=topic_counts,
            n_runs=held_out_runs,
            base_random_seed=base_random_seed,
            test_size=test_size,
            doc_topic_prior=doc_topic_prior,
            topic_word_prior=topic_word_prior,
        )
    return compute_in_sample_perplexity(
        texts=texts,
        stop_words=stop_words,
        topic_counts=topic_counts,
        random_seed=base_random_seed,
        doc_topic_prior=doc_topic_prior,
        topic_word_prior=topic_word_prior,
    )
