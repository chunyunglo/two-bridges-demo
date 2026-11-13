"""The frozen scoring pipeline (the composite A*) and the statistics used to evaluate it."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .routes import Anchors, Distributions, fidelity, rc, rc_positive_max, self_consistency, self_fidelity
from .spaces import Space, half_sample

COMPONENTS = ["rc_u_star", "rc_v", "f_fwd_star", "f_bwd"]


class TwoBridges:
    """Scores cross-language word pairs from two association data sets and a bilingual lexicon.

    Parameters
    ----------
    en, zh : long tables with the columns pid, cue, response (one row per response token)
    lexicon : table with the columns English, Chinese (one row per translation pair)
    rank : dimensionality of the association spaces
    min_count : a cue or response is kept if it occurs at least this often
    n_splits : participant split-half replications behind the within-language ceilings
    floor : lower bound on a ceiling before it is used as a divisor

    Building the object does all the work that needs no pair: spaces, anchors, association
    distributions, per-word ceilings, and the means and standard deviations of every component
    over the lexicon pairs. `score()` then applies those frozen constants to any pairs.
    """

    def __init__(self, en: pd.DataFrame, zh: pd.DataFrame, lexicon: pd.DataFrame, rank: int = 40,
                 min_count: int = 5, n_splits: int = 3, floor: float = 0.3, seed: int = 0):
        self.lexicon = lexicon[["English", "Chinese"]].drop_duplicates().reset_index(drop=True)
        self.floor = floor
        long = {"en": en, "zh": zh}
        full = {lang: Space(long[lang], min_count, rank) for lang in long}
        self.views, self.anchors = {}, {}
        for role in ("U", "V"):
            (ew, ee), (zw, ze) = full["en"].view(role), full["zh"].view(role)
            self.views[role] = (ew, ee, zw, ze)
            self.anchors[role] = Anchors(self.lexicon, ew, zw)
        self.dist = {role: {lang: Distributions(long[lang], role, min_count) for lang in long} for role in ("fwd", "bwd")}

        # Within-language ceilings: how well does a word agree with itself across speakers?
        half_min = max(1, round(0.6 * min_count))
        anchor_words = {"en": self.anchors["U"].en_words, "zh": np.unique(self.anchors["U"].zh_words)}
        rel_rc, rel_f = {"en": [], "zh": []}, {"en": [], "zh": []}
        for s in range(n_splits):
            for lang in long:
                a, b = (half_sample(long[lang], s, w, seed) for w in "AB")
                (wa, ea), (wb, eb) = Space(a, half_min, rank).view("U"), Space(b, half_min, rank).view("U")
                rel_rc[lang].append(self_consistency(wa, ea, wb, eb, anchor_words[lang]))
                d = self.dist["fwd"][lang]
                rel_f[lang].append(self_fidelity(*(Distributions(h, "fwd", 1, words=d.words, features=d.features)
                                                   for h in (a, b))))
        self.ceiling_rc = {lang: pd.concat(rel_rc[lang], axis=1).mean(axis=1) for lang in long}
        self.ceiling_f = {lang: pd.concat(rel_f[lang], axis=1).mean(axis=1) for lang in long}

        population = self._components(self.lexicon)
        self.mean, self.sd = population[COMPONENTS + ["rc_u", "f_fwd"]].mean(), population[COMPONENTS + ["rc_u", "f_fwd"]].std()

    def _components(self, pairs: pd.DataFrame, drop_self: bool = False) -> pd.DataFrame:
        out = pairs[["English", "Chinese"]].reset_index(drop=True).copy()
        for role in ("U", "V"):
            out[f"rc_{role.lower()}"] = rc(out, *self.views[role], self.anchors[role])
        out["rc_baseline"] = rc_positive_max(out, *self.views["U"], self.anchors["U"])
        for role in ("fwd", "bwd"):
            out[f"f_{role}"] = fidelity(out, self.dist[role]["en"], self.dist[role]["zh"], self.lexicon, drop_self=drop_self)
        for name, table in (("rc", self.ceiling_rc), ("f", self.ceiling_f)):
            out[f"ceiling_{name}"] = np.sqrt(out.English.map(table["en"]).clip(lower=self.floor)
                                             * out.Chinese.map(table["zh"]).clip(lower=self.floor))
        out["rc_u_star"] = out.rc_u / out.ceiling_rc
        out["f_fwd_star"] = out.f_fwd / out.ceiling_f
        return out

    def score(self, pairs: pd.DataFrame | None = None, drop_self: bool = False) -> pd.DataFrame:
        """Component scores and composites for `pairs` (default: every lexicon pair).

        rc_baseline   the earlier definition of relational consistency (cue role)
        rc_u, rc_v    relational consistency over all anchors, cue role / response role
        f_fwd, f_bwd  fidelity of translated association distributions, cue / response role
        rc_star       second-order route (ceiling-corrected cue role + response role)
        f_star        first-order route  (ceiling-corrected cue role + response role)
        a_star        both routes: the mean of the available z-scores of the four components
        a_raw         the same without the ceiling correction
        """
        out = self._components(self.lexicon if pairs is None else pairs, drop_self)
        z = lambda c: (out[c] - self.mean[c]) / self.sd[c]
        out["rc_uv"] = pd.concat([z("rc_u"), z("rc_v")], axis=1).mean(axis=1)
        out["rc_star"] = pd.concat([z("rc_u_star"), z("rc_v")], axis=1).mean(axis=1)
        out["f_star"] = pd.concat([z("f_fwd_star"), z("f_bwd")], axis=1).mean(axis=1)
        out["a_star"] = pd.concat([z(c) for c in COMPONENTS], axis=1).mean(axis=1)
        out["a_raw"] = pd.concat([z("rc_u"), z("rc_v"), z("f_fwd"), z("f_bwd")], axis=1).mean(axis=1)
        return out


# ------------------------------------------------------------------------ statistics
def split_of(word: str) -> str:
    """Fixed development/test split by a hash of the English word, made before any modelling."""
    return "dev" if int(hashlib.sha1(word.encode("utf8")).hexdigest(), 16) % 2 == 0 else "test"


def spearman_cols(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman correlation of every column of `scores` with y (no missing values)."""
    s = np.apply_along_axis(rankdata, 0, scores)
    r = rankdata(y)
    s, r = s - s.mean(0), r - r.mean()
    return (s * r[:, None]).sum(0) / np.sqrt((s * s).sum(0) * (r @ r))


def evaluate(frame: pd.DataFrame, cols: list[str], target: str, baseline: str = "rc_baseline",
             cluster: str = "English", n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Spearman correlation of each score with `target` on the rows where every score exists,
    with percentile intervals from a bootstrap that resamples whole clusters (English words),
    and the paired difference of each score from the baseline."""
    frame = frame.dropna(subset=cols + [target]).reset_index(drop=True)
    s, y = frame[cols].to_numpy(float), frame[target].to_numpy(float)
    groups = [np.asarray(g) for g in frame.groupby(cluster).indices.values()]
    rng = np.random.default_rng(seed)
    reps = np.empty((n_boot, len(cols)))
    for b in range(n_boot):
        take = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        reps[b] = spearman_cols(s[take], y[take])
    point, base = spearman_cols(s, y), cols.index(baseline)
    lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
    dlo, dhi = np.percentile(reps - reps[:, [base]], [2.5, 97.5], axis=0)
    return pd.DataFrame({"score": cols, "target": target, "n": len(frame), "rho": point, "lo": lo, "hi": hi,
                         "diff": point - point[base], "diff_lo": dlo, "diff_hi": dhi})
