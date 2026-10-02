import math
from collections import Counter
import numpy as np

def bm25_score(
    query_tokens: list[str],
    docs: list[list[str]],
    k1: float = 1.2,
    b: float = 0.75
) -> np.ndarray:
    """
    Returns a NumPy array with one BM25 score per document.
    """
    N = len(docs)

    if N == 0:
        return np.array([], dtype=float)

    avgdl = sum(len(doc) for doc in docs) / N

    query_terms = set(query_tokens)

    doc_counts = [Counter(doc) for doc in docs]

    df = {
        term: sum(1 for counts in doc_counts if counts[term] > 0)
        for term in query_terms
    }

    idf = {
        term: math.log(
            (N - df[term] + 0.5) / (df[term] + 0.5) + 1
        )
        for term in query_terms
    }

    scores = np.zeros(N, dtype=float)

    if avgdl == 0:
        return scores

    for i, (doc, counts) in enumerate(zip(docs, doc_counts)):
        doc_len = len(doc)

        for term in query_terms:
            tf = counts[term]

            if tf == 0:
                continue

            denominator = (
                tf
                + k1 * (1 - b + b * doc_len / avgdl)
            )

            scores[i] += (
                idf[term]
                * tf
                * (k1 + 1)
                / denominator
            )

    return scores