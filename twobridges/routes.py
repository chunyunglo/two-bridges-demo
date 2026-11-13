"""The two lexicon-bridged routes between a word in one language and a word in the other.

Second-order route, relational consistency `rc`: do the two words stand in the same relations
to a set of translated anchor words? Only within-language similarities are used, so the two
spaces never have to be aligned.

First-order route, translated-association `fidelity`: push one word's association distribution
through the bilingual lexicon and compare it with the other word's distribution.

The column names `English` and `Chinese` come from the original project; any two languages work.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from .spaces import count_matrix


# ---------------------------------------------------------------- second-order route
class Anchors:
    """Translation-pair anchors grouped by English word.

    `en_idx[g]` is the row of English anchor g; the rows j with `group[j] == g` are its
    translations `zh_idx[j]`. Rows are sorted by group so `reduceat` works.
    """

    def __init__(self, lexicon: pd.DataFrame, en_words, zh_words):
        e_pos = pd.Series(np.arange(len(en_words)), index=en_words)
        z_pos = pd.Series(np.arange(len(zh_words)), index=zh_words)
        d = lexicon[lexicon.English.isin(e_pos.index) & lexicon.Chinese.isin(z_pos.index)].drop_duplicates()
        d = d.sort_values(["English", "Chinese"]).reset_index(drop=True)
        self.pairs = d
        self.en_words = d.English.drop_duplicates().to_numpy()
        self.en_idx = e_pos[self.en_words].to_numpy()
        self.group = pd.Series(np.arange(len(self.en_words)), index=self.en_words)[d.English].to_numpy()
        self.zh_words = d.Chinese.to_numpy()
        self.zh_idx = z_pos[self.zh_words].to_numpy()
        self.starts = np.flatnonzero(np.r_[True, np.diff(self.group) > 0])
        self.en_pos = {w: g for g, w in enumerate(self.en_words)}
        self.zh_rows = {w: rows.to_numpy() for w, rows in pd.Series(np.arange(len(d))).groupby(self.zh_words)}


def pearson(x, y) -> float:
    if len(x) < 3:
        return np.nan
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt((x @ x) * (y @ y))
    return float(x @ y / den) if den > 0 else np.nan


def masked_corr(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Column-wise Pearson correlation over the rows where mask is True."""
    m = mask.astype(np.float64)
    n = np.maximum(m.sum(0), 1)
    xm, ym = (x * m).sum(0) / n, (y * m).sum(0) / n
    xc, yc = (x - xm) * m, (y - ym) * m
    den = np.sqrt((xc * xc).sum(0) * (yc * yc).sum(0))
    return np.where(den > 0, (xc * yc).sum(0) / np.where(den > 0, den, 1), np.nan)


def rc(pairs: pd.DataFrame, en_words, en_emb, zh_words, zh_emb, anchors: Anchors, batch: int = 512) -> np.ndarray:
    """Relational consistency over all anchors.

    For the pair (e, c): x_g = cos(e, anchor_g) and y_g = the mean, over the translations of
    anchor_g, of cos(c, translation). Anchor pairs that contain e or c are left out. Rc is the
    Pearson correlation of x and y over the anchors g.
    """
    e_pos = {w: i for i, w in enumerate(en_words)}
    z_pos = {w: i for i, w in enumerate(zh_words)}
    en_a, zh_a = en_emb[anchors.en_idx], zh_emb[anchors.zh_idx]
    size = np.diff(np.r_[anchors.starts, len(anchors.group)]).astype(np.float64)
    pe, pc = pairs.English.to_numpy(), pairs.Chinese.to_numpy()
    have = np.array([e in e_pos and c in z_pos for e, c in zip(pe, pc)], dtype=bool)
    out = np.full(len(pairs), np.nan)
    todo = np.flatnonzero(have)
    for start in range(0, len(todo), batch):
        idx = todo[start:start + batch]
        x = (en_a @ en_emb[[e_pos[w] for w in pe[idx]]].T).astype(np.float64)       # anchors x B
        y_rows = (zh_a @ zh_emb[[z_pos[w] for w in pc[idx]]].T).astype(np.float64)  # anchor translations x B
        y_sum = np.add.reduceat(y_rows, anchors.starts, axis=0)
        cnt = np.repeat(size[:, None], len(idx), axis=1)
        for b, i in enumerate(idx):
            rows = anchors.zh_rows.get(pc[i])
            if rows is not None:                       # the focal word is not one of an anchor's translations
                np.subtract.at(y_sum[:, b], anchors.group[rows], y_rows[rows, b])
                np.subtract.at(cnt[:, b], anchors.group[rows], 1.0)
            g = anchors.en_pos.get(pe[i])
            if g is not None:                          # the focal word is not its own anchor
                cnt[g, b] = 0.0
        out[idx] = masked_corr(x, y_sum / np.maximum(cnt, 1.0), cnt > 0)
    return out


def rc_positive_max(pairs: pd.DataFrame, en_words, en_emb, zh_words, zh_emb, anchors: Anchors) -> np.ndarray:
    """The baseline definition: only anchors with positive similarity on both sides, and the
    best-matching translation of each anchor instead of the mean."""
    e_pos = {w: i for i, w in enumerate(en_words)}
    z_pos = {w: i for i, w in enumerate(zh_words)}
    en_a, zh_a = en_emb[anchors.en_idx], zh_emb[anchors.zh_idx]
    out = np.full(len(pairs), np.nan)
    for i, (e, c) in enumerate(zip(pairs.English, pairs.Chinese)):
        if e not in e_pos or c not in z_pos:
            continue
        x_all = en_a @ en_emb[e_pos[e]]
        y_rows = zh_a @ zh_emb[z_pos[c]]
        ok = (anchors.en_words[anchors.group] != e) & (anchors.zh_words != c)
        ok &= (x_all[anchors.group] > 0) & (y_rows > 0)
        y = np.maximum.reduceat(np.where(ok, y_rows, -np.inf), anchors.starts)
        keep = np.isfinite(y)
        out[i] = pearson(x_all[keep].astype(np.float64), y[keep].astype(np.float64))
    return out


def self_consistency(words_a, emb_a, words_b, emb_b, anchor_words, batch: int = 1024) -> pd.Series:
    """Relational consistency of each word with itself across two halves of the participants.

    The anchor profile of a word is computed separately in half A and in half B, each in its
    own space, and the two profiles are correlated. This is the most a cross-language Rc could
    reach if translation lost nothing: a within-language ceiling.
    """
    ia = {w: i for i, w in enumerate(words_a)}
    ib = {w: i for i, w in enumerate(words_b)}
    anchors = np.array([w for w in anchor_words if w in ia and w in ib])
    a_pos = {w: i for i, w in enumerate(anchors)}
    anc_a, anc_b = emb_a[[ia[w] for w in anchors]], emb_b[[ib[w] for w in anchors]]
    words = np.array([w for w in words_a if w in ib])
    out = np.empty(len(words))
    for start in range(0, len(words), batch):
        chunk = words[start:start + batch]
        x = (anc_a @ emb_a[[ia[w] for w in chunk]].T).astype(np.float64)
        y = (anc_b @ emb_b[[ib[w] for w in chunk]].T).astype(np.float64)
        mask = np.ones(x.shape, dtype=bool)
        for b, w in enumerate(chunk):
            if w in a_pos:
                mask[a_pos[w], b] = False
        out[start:start + batch] = masked_corr(x, y, mask)
    return pd.Series(out, index=words)


def procrustes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Orthogonal W minimising ||source @ W - target||_F (only used to show it is redundant)."""
    u, _, vt = np.linalg.svd(target.T.astype(np.float64) @ source.astype(np.float64))
    return vt.T @ u.T


# ----------------------------------------------------------------- first-order route
class Distributions:
    """Row-stochastic association distributions of one language in one role.

    role 'fwd': a word is described by what it evokes       (rows = cues, features = responses)
    role 'bwd': a word is described by what evokes it       (rows = responses, features = cues)
    One step of spreading activation is mixed in: a feature that is itself a row word passes on
    its own distribution (associates of associates).
    """

    def __init__(self, long: pd.DataFrame, role: str, min_count: int = 5, diffusion: float = 1.0,
                 words=None, features=None):
        counts, rows, feats = count_matrix(long, min_count, min_count)
        if role == "bwd":
            counts, rows, feats = counts.T.tocsr(), feats, rows
        if words is not None:                      # re-index onto a fixed vocabulary (participant halves)
            rpos, fpos = {w: i for i, w in enumerate(words)}, {w: i for i, w in enumerate(features)}
            coo = counts.tocoo()
            r = np.array([rpos.get(rows[i], -1) for i in coo.row])
            f = np.array([fpos.get(feats[j], -1) for j in coo.col])
            keep = (r >= 0) & (f >= 0)
            counts = csr_matrix((coo.data[keep], (r[keep], f[keep])), shape=(len(words), len(features)))
            rows, feats = np.asarray(words), np.asarray(features)
        self.words, self.features = rows, feats
        self.pos = {w: i for i, w in enumerate(rows)}
        fpos = {w: j for j, w in enumerate(feats)}
        self.base = normalize(counts, norm="l1").tocsr().astype(np.float32)
        self.diffusion = diffusion
        both = [w for w in rows if w in fpos]
        hop = csr_matrix((np.ones(len(both), dtype=np.float32), ([fpos[w] for w in both], [self.pos[w] for w in both])),
                         shape=(len(feats), len(rows)))
        self.step = (hop @ self.base).tocsr()      # feature word -> that word's own distribution

    def rows(self, idx):
        """Diffused, renormalised distributions for the given row indices."""
        p = self.base[idx]
        if self.diffusion > 0:
            p = normalize(p + self.diffusion * normalize(p @ self.step, norm="l1"), norm="l1")
        return p


def lexicon_push(lexicon: pd.DataFrame, en_features, zh_features):
    """Row-stochastic lexicon couplings between two feature vocabularies (en->zh, zh->en)."""
    ei, zi = {w: i for i, w in enumerate(en_features)}, {w: i for i, w in enumerate(zh_features)}
    ok = lexicon[lexicon.English.isin(ei) & lexicon.Chinese.isin(zi)]
    t = csr_matrix((np.ones(len(ok), dtype=np.float32), ([ei[w] for w in ok.English], [zi[w] for w in ok.Chinese])),
                   shape=(len(en_features), len(zh_features)))
    return normalize(t, norm="l1"), normalize(t.T.tocsr(), norm="l1")


def _without(m, cols):
    """Rows of sparse m with the entry in column cols[i] removed (cols[i] < 0: nothing to
    remove), renormalised to sum to one."""
    rows = np.flatnonzero(cols >= 0)
    mask = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols[rows])), shape=m.shape)
    return normalize(m - m.multiply(mask), norm="l1")


def fidelity(pairs: pd.DataFrame, en: Distributions, zh: Distributions, lexicon: pd.DataFrame, batch: int = 1024,
             drop_self: bool = False) -> np.ndarray:
    """Translated-association fidelity F(e, c).

    p(.|e) is pushed through the lexicon into the other language's feature space, renormalised,
    and compared with p(.|c) by the Bhattacharyya coefficient sum_r sqrt(p q), the cosine of the
    Fisher-Rao angle between two distributions. The same is done from c towards e and the two
    directions are averaged.

    drop_self=True removes each word from its own distribution first (after one step of
    spreading activation a word is among its own associates), so that a lexicon entry (e, c)
    cannot contribute to its own score. Use it when only some of the scored pairs are lexicon
    entries.
    """
    if drop_self:
        fe, fz = {w: j for j, w in enumerate(en.features)}, {w: j for j, w in enumerate(zh.features)}
        self_e = np.array([fe.get(w, -1) for w in en.words])
        self_z = np.array([fz.get(w, -1) for w in zh.words])
    en2zh, zh2en = lexicon_push(lexicon, en.features, zh.features)
    pe, pc = pairs.English.to_numpy(), pairs.Chinese.to_numpy()
    have = np.array([e in en.pos and c in zh.pos for e, c in zip(pe, pc)], dtype=bool)
    out = np.full(len(pairs), np.nan)
    ie = np.array([en.pos[w] for w in pe[have]], dtype=int)
    iz = np.array([zh.pos[w] for w in pc[have]], dtype=int)
    total = np.zeros(int(have.sum()))
    for start in range(0, len(ie), batch):
        sel = np.arange(start, min(start + batch, len(ie)))
        p, q = en.rows(ie[sel]), zh.rows(iz[sel])
        if drop_self:
            p, q = _without(p, self_e[ie[sel]]), _without(q, self_z[iz[sel]])
        fwd = normalize(p @ en2zh, norm="l1").sqrt().multiply(q.sqrt()).sum(axis=1)
        bwd = normalize(q @ zh2en, norm="l1").sqrt().multiply(p.sqrt()).sum(axis=1)
        total[sel] = (np.asarray(fwd).ravel() + np.asarray(bwd).ravel()) / 2
    out[have] = total
    return out


def self_fidelity(a: Distributions, b: Distributions, batch: int = 1024) -> pd.Series:
    """Bhattacharyya coefficient between a word's distributions in two participant halves."""
    assert (a.words == b.words).all() and (a.features == b.features).all()
    out = np.empty(len(a.words))
    for start in range(0, len(out), batch):
        idx = np.arange(start, min(start + batch, len(out)))
        out[idx] = np.asarray(a.rows(idx).sqrt().multiply(b.rows(idx).sqrt()).sum(axis=1)).ravel()
    return pd.Series(out, index=a.words)
