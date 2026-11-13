"""A synthetic bilingual world whose ground truth is known.

Nothing here is real: the "words" are labels such as en_0412 and zh_0412, the "participants"
are random draws, and the "ratings" are produced by simulated raters. The point of the
simulation is that the quantity the method tries to estimate, how close the meanings of two
words are, is known exactly, so one can check whether the method recovers it.

How the world is built
  1. Concepts are points in a latent meaning space, grouped into topics.
  2. Each concept has an English word. Its translation into the second language sits at the
     same point plus a drift whose size varies from pair to pair: a little for close
     equivalents, more for loose ones. Some English words have a second translation, and some
     dictionary entries pair a word with the translation of a neighbouring concept.
  3. In each language, simulated participants see a cue and give three different responses.
     A response is more likely the closer it is to the cue in that language's own meaning
     space and the more frequent it is.
  4. Simulated raters judge the similarity of a sample of dictionary pairs on a 1-7 scale.

The true equivalence of a pair is the cosine between its two latent points. The method never
sees the latent space; it sees only the response tokens and the dictionary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .pipeline import split_of


@dataclass(frozen=True)
class WorldConfig:
    n_concepts: int = 1200        # English vocabulary size
    n_topics: int = 30
    dim: int = 32                 # dimensionality of the latent meaning space
    spread: float = 0.9           # how far concepts scatter around their topic
    second_rate: float = 0.25     # share of English words with a second translation
    loose_rate: float = 0.15      # share of English words with an extra, loosely fitting entry
    cue_rate: float = 0.85        # share of words presented as cues (the rest occur only as responses)
    kappa: float = 10.0           # how sharply responses follow similarity to the cue
    freq_weight: float = 0.5      # how strongly frequent words attract responses
    participants: int = 100       # participants per cue; each gives three responses
    cues_per_participant: int = 16
    rated_share: float = 0.5      # share of dictionary pairs that receive ratings
    raters: int = 10
    rater_sd: float = 1.2


@dataclass
class World:
    config: WorldConfig
    en_words: np.ndarray
    zh_words: np.ndarray
    en_vec: np.ndarray
    zh_vec: np.ndarray
    lexicon: pd.DataFrame          # English, Chinese, kind, truth
    ratings: pd.DataFrame          # English, Chinese, rating, truth, split
    _long: dict = field(repr=False, default_factory=dict)

    def associations(self, lang: str, participants: int | None = None) -> pd.DataFrame:
        """Response tokens (pid, cue, response) from the first `participants` participants per cue."""
        long = self._long[lang]
        if participants is not None:
            long = long[long.slot < participants]
        return long[["pid", "cue", "response"]].reset_index(drop=True)

    def truth(self, pairs: pd.DataFrame) -> np.ndarray:
        """True equivalence (latent cosine) of arbitrary cross-language pairs."""
        e = pd.Series(np.arange(len(self.en_words)), index=self.en_words)[pairs.English].to_numpy()
        z = pd.Series(np.arange(len(self.zh_words)), index=self.zh_words)[pairs.Chinese].to_numpy()
        return np.einsum("ij,ij->i", self.en_vec[e], self.zh_vec[z])

    def probe_pairs(self, n: int = 1500, seed: int = 0) -> pd.DataFrame:
        """Rated cross-language pairs that are not dictionary entries, from near to unrelated:
        half pair a word with a translation of one of its ten nearest concepts, half are random."""
        rng = np.random.default_rng(seed)
        sim = self.en_vec @ self.en_vec.T
        np.fill_diagonal(sim, -np.inf)
        near = np.argsort(-sim, axis=1)[:, :10]
        e = rng.integers(0, len(self.en_words), n)
        other = np.where(rng.random(n) < 0.5, near[e, rng.integers(0, 10, n)], rng.integers(0, len(self.en_words), n))
        pairs = pd.DataFrame({"English": self.en_words[e], "Chinese": self.zh_words[other]}).drop_duplicates()
        entries = set(zip(self.lexicon.English, self.lexicon.Chinese))
        pairs = pairs[[(a, b) not in entries for a, b in zip(pairs.English, pairs.Chinese)]].reset_index(drop=True)
        pairs["truth"] = self.truth(pairs)
        pairs["rating"] = _rate(pairs.truth.to_numpy(), self.config, rng)
        return pairs


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _rate(truth: np.ndarray, cfg: WorldConfig, rng) -> np.ndarray:
    """Mean of `raters` noisy integer judgements on a 1-7 scale."""
    ideal = 1 + 6 * np.clip(truth, 0, 1) ** 2
    votes = np.rint(ideal[:, None] + rng.normal(0, cfg.rater_sd, (len(truth), cfg.raters)))
    return np.clip(votes, 1, 7).mean(axis=1)


def _associate(vec: np.ndarray, words: np.ndarray, cfg: WorldConfig, rng) -> pd.DataFrame:
    """Simulated free association in one language: three distinct responses per participant and cue."""
    n = len(words)
    freq = 1.0 / rng.permutation(np.arange(1, n + 1))                  # Zipfian word frequencies
    logit = (cfg.kappa * (vec @ vec.T) + cfg.freq_weight * np.log(freq)[None, :]).astype(np.float32)
    np.fill_diagonal(logit, -np.inf)                                   # a cue is never its own response
    cues = np.sort(rng.choice(n, int(round(cfg.cue_rate * n)), replace=False))
    blocks = int(np.ceil(len(cues) / cfg.cues_per_participant))        # participants per slot
    pid = np.stack([s * blocks + rng.permutation(len(cues)) // cfg.cues_per_participant
                    for s in range(cfg.participants)])                 # slot x cue -> participant id
    responses = np.empty((len(cues), cfg.participants, 3), dtype=np.int32)
    for k, c in enumerate(cues):                                       # Gumbel top-3 = sampling without replacement
        noisy = logit[c] + rng.gumbel(size=(cfg.participants, n)).astype(np.float32)
        responses[k] = np.argpartition(-noisy, 3, axis=1)[:, :3]
    slot = np.broadcast_to(np.arange(cfg.participants)[None, :, None], responses.shape)
    cue = np.broadcast_to(cues[:, None, None], responses.shape)
    return pd.DataFrame({"pid": np.broadcast_to(pid.T[:, :, None], responses.shape).ravel().astype(np.int32),
                         "slot": slot.ravel().astype(np.int16),
                         "cue": words[cue.ravel()], "response": words[responses.ravel()]})


def make_world(config: WorldConfig = WorldConfig(), seed: int = 0) -> World:
    cfg, rng = config, np.random.default_rng(seed)
    n, d = cfg.n_concepts, cfg.dim
    noise = lambda rows: rng.normal(size=(rows, d)) / np.sqrt(d)       # random directions of length about one

    topic = rng.integers(0, cfg.n_topics, n)
    concept = _unit(noise(cfg.n_topics)[topic] + cfg.spread * noise(n))
    en_words = np.array([f"en_{i:04d}" for i in range(n)])

    # First translations: most are close equivalents, some have drifted.
    drift = 1.5 * rng.beta(1.2, 3.0, n)
    zh_vec = [_unit(concept + drift[:, None] * noise(n))]
    zh_words = [f"zh_{i:04d}" for i in range(n)]
    entries = [(i, i, "first") for i in range(n)]
    # Second translations of some words, on average less close.
    second = np.sort(rng.choice(n, int(round(cfg.second_rate * n)), replace=False))
    zh_vec.append(_unit(concept[second] + (1.5 * rng.beta(2.0, 2.0, len(second)))[:, None] * noise(len(second))))
    zh_words += [f"zh_{i:04d}b" for i in second]
    entries += [(i, n + k, "second") for k, i in enumerate(second)]
    # Loose entries: a word listed with the translation of one of its three nearest concepts.
    sim = concept @ concept.T
    np.fill_diagonal(sim, -np.inf)
    nearest = np.argsort(-sim, axis=1)[:, :3]
    loose = np.sort(rng.choice(n, int(round(cfg.loose_rate * n)), replace=False))
    entries += [(i, int(nearest[i, rng.integers(0, 3)]), "loose") for i in loose]

    zh_vec, zh_words = np.vstack(zh_vec), np.array(zh_words)
    world = World(cfg, en_words, zh_words, concept, zh_vec, pd.DataFrame(), pd.DataFrame())
    lexicon = pd.DataFrame({"English": en_words[[e for e, _, _ in entries]], "Chinese": zh_words[[z for _, z, _ in entries]],
                            "kind": [k for _, _, k in entries]})
    lexicon["truth"] = world.truth(lexicon)
    world.lexicon = lexicon

    rated = lexicon.sample(frac=cfg.rated_share, random_state=seed).sort_index().reset_index(drop=True)
    rated["rating"] = _rate(rated.truth.to_numpy(), cfg, rng)
    rated["split"] = rated.English.map(split_of)
    world.ratings = rated[["English", "Chinese", "kind", "rating", "truth", "split"]]

    world._long = {"en": _associate(concept, en_words, cfg, rng), "zh": _associate(zh_vec, zh_words, cfg, rng)}
    return world
