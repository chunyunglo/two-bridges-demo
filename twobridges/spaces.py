"""Association spaces: from (cue, response) tokens to cue-role and response-role vectors.

A free-association data set is a long table with one row per response token and the columns
`pid` (participant), `cue` and `response`. The cue x response count matrix is reweighted by
positive pointwise mutual information and factorised; a word then has two vectors, one for what
it evokes when it is a cue (role "U") and one for what evokes it as a response (role "V").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import svds


def count_matrix(long: pd.DataFrame, min_cue: int = 5, min_response: int = 5):
    """Directed cue x response count matrix. Returns (csr, cue_words, response_words)."""
    d = long[["cue", "response"]]
    cue_n, resp_n = d.cue.value_counts(), d.response.value_counts()
    d = d[d.cue.isin(cue_n.index[cue_n >= min_cue]) & d.response.isin(resp_n.index[resp_n >= min_response])]
    cues, responses = np.array(sorted(d.cue.unique())), np.array(sorted(d.response.unique()))
    ci = pd.Series(np.arange(len(cues)), index=cues)
    ri = pd.Series(np.arange(len(responses)), index=responses)
    mat = csr_matrix((np.ones(len(d)), (ci[d.cue].to_numpy(), ri[d.response].to_numpy())),
                     shape=(len(cues), len(responses)))
    mat.sum_duplicates()
    return mat, cues, responses


def ppmi(counts: csr_matrix) -> csr_matrix:
    """Positive pointwise mutual information of a count matrix."""
    total = counts.sum()
    rows = np.asarray(counts.sum(axis=1)).ravel()
    cols = np.asarray(counts.sum(axis=0)).ravel()
    rows[rows == 0] = 1
    cols[cols == 0] = 1
    out = (total * (diags(1 / rows) @ counts @ diags(1 / cols))).tocsr()
    out.data = np.maximum(np.log(out.data), 0)
    out.eliminate_zeros()
    return out


def l2norm(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1
    return (x / denom).astype(np.float32)


def svd_factors(matrix: csr_matrix, k: int):
    """Top-k singular triplets by decreasing singular value (fixed start vector: deterministic)."""
    v0 = np.ones(min(matrix.shape)) / np.sqrt(min(matrix.shape))
    u, s, vt = svds(matrix.astype(np.float64), k=k, v0=v0)
    order = np.argsort(s)[::-1]
    return u[:, order], s[order], vt[order].T


class Space:
    """One language's association space: cue-role (U) and response-role (V) vectors."""

    def __init__(self, long: pd.DataFrame, min_count: int = 5, rank: int = 40):
        counts, self.cues, self.responses = count_matrix(long, min_count, min_count)
        self.u, self.s, self.v = svd_factors(ppmi(counts), min(rank, min(counts.shape) - 1))

    def view(self, role: str, power: float = 0.5):
        """(words, unit vectors) for role 'U' (word as cue) or 'V' (word as response)."""
        words, factor = (self.cues, self.u) if role == "U" else (self.responses, self.v)
        return words, l2norm(factor * self.s ** power)


def half_sample(long: pd.DataFrame, split: int, which: str, seed: int = 0) -> pd.DataFrame:
    """Responses of one random half of the participants (`split` fixes the partition)."""
    pids = np.sort(long.pid.unique())
    np.random.default_rng(seed + split).shuffle(pids)
    in_a = long.pid.isin(pids[: len(pids) // 2])
    return long[in_a if which == "A" else ~in_a]
