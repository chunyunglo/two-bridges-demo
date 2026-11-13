"""The two routes on small random inputs:  python -m pytest -q"""
import numpy as np
import pandas as pd

from twobridges.routes import (Anchors, Distributions, fidelity, masked_corr, pearson, procrustes, rc,
                               rc_positive_max, self_fidelity)
from twobridges.spaces import l2norm

rng = np.random.default_rng(0)
N_EN, N_ZH, K = 60, 70, 12
en_words = np.array([f"e{i}" for i in range(N_EN)])
zh_words = np.array([f"z{i}" for i in range(N_ZH)])
en, zh = l2norm(rng.normal(size=(N_EN, K))), l2norm(rng.normal(size=(N_ZH, K)))
lex = pd.DataFrame({"English": rng.choice(en_words, 150), "Chinese": rng.choice(zh_words, 150)}).drop_duplicates()
anchors = Anchors(lex, en_words, zh_words)
pairs = lex.head(40).reset_index(drop=True)


def naive_rc(e, c):
    """Direct transcription of the definition, one anchor at a time."""
    xs, ys = [], []
    for a in sorted(set(lex.English)):
        if a == e:
            continue
        trans = [t for t in lex[lex.English == a].Chinese if t != c]
        if not trans:
            continue
        xs.append(float(en[np.flatnonzero(en_words == a)[0]] @ en[np.flatnonzero(en_words == e)[0]]))
        ys.append(np.mean([float(zh[np.flatnonzero(zh_words == t)[0]] @ zh[np.flatnonzero(zh_words == c)[0]]) for t in trans]))
    return pearson(np.array(xs), np.array(ys))


def test_batched_rc_matches_definition():
    fast = rc(pairs, en_words, en, zh_words, zh, anchors, batch=7)
    slow = np.array([naive_rc(e, c) for e, c in zip(pairs.English, pairs.Chinese)])
    assert np.allclose(fast, slow, atol=1e-6)


def test_rc_is_invariant_to_orthogonal_maps():
    """Rotating one language's space changes nothing: an explicit alignment step is redundant."""
    q, _ = np.linalg.qr(rng.normal(size=(K, K)))
    base = rc(pairs, en_words, en, zh_words, zh, anchors)
    assert np.allclose(base, rc(pairs, en_words, en, zh_words, l2norm(zh @ q), anchors), atol=1e-5)
    assert np.allclose(rc_positive_max(pairs, en_words, en, zh_words, zh, anchors),
                       rc_positive_max(pairs, en_words, en, zh_words, l2norm(zh @ q), anchors), atol=1e-5, equal_nan=True)


def test_procrustes_is_orthogonal_and_recovers_rotation():
    q, _ = np.linalg.qr(rng.normal(size=(K, K)))
    w = procrustes(en @ q, en)
    assert np.allclose(w.T @ w, np.eye(K), atol=1e-8)
    assert np.allclose(en @ q @ w, en, atol=1e-5)


def test_rc_equals_bilinear_form_without_leave_out():
    """With anchors disjoint from the scored words, Rc = e'Mc / sqrt(e'G_e e c'G_c c)."""
    held_e, held_z = en_words[:10], zh_words[:10]
    free = lex[~lex.English.isin(held_e) & ~lex.Chinese.isin(held_z)]
    fa = Anchors(free, en_words, zh_words)
    probe = pd.DataFrame({"English": held_e, "Chinese": held_z})
    explicit = rc(probe, en_words, en, zh_words, zh, fa)
    a_en = en[fa.en_idx].astype(float)
    size = np.diff(np.r_[fa.starts, len(fa.group)])[:, None]
    a_zh = np.add.reduceat(zh[fa.zh_idx].astype(float), fa.starts, axis=0) / size
    ac, zc = a_en - a_en.mean(0), a_zh - a_zh.mean(0)
    m, ge, gz = ac.T @ zc, ac.T @ ac, zc.T @ zc
    e, z = en[:10].astype(float), zh[:10].astype(float)
    form = np.einsum("ij,jk,ik->i", e, m, z) / np.sqrt(np.einsum("ij,jk,ik->i", e, ge, e) * np.einsum("ij,jk,ik->i", z, gz, z))
    assert np.allclose(explicit, form, atol=1e-6)


def test_masked_corr():
    x, y = rng.normal(size=(30, 3)), rng.normal(size=(30, 3))
    mask = rng.random((30, 3)) > 0.2
    out = masked_corr(x, y, mask)
    for j in range(3):
        assert np.isclose(out[j], np.corrcoef(x[mask[:, j], j], y[mask[:, j], j])[0, 1])


def _tokens(words, n=4000, seed=1):
    r = np.random.default_rng(seed)
    return pd.DataFrame({"pid": r.integers(0, 50, n), "cue": r.choice(words, n), "response": r.choice(words, n)})


def test_fidelity_is_one_for_a_language_and_its_relabelled_copy():
    """If the second language is the first with every word renamed, and the lexicon is that
    renaming, each word is a perfect equivalent of its copy and of nothing else."""
    words = np.array([f"w{i}" for i in range(30)])
    a = _tokens(words)
    b = a.assign(cue="x" + a.cue, response="x" + a.response)
    lexicon = pd.DataFrame({"English": words, "Chinese": ["x" + w for w in words]})
    for role in ("fwd", "bwd"):
        da, db = Distributions(a, role, min_count=1), Distributions(b, role, min_count=1)
        same = fidelity(lexicon, da, db, lexicon)
        other = fidelity(lexicon.assign(Chinese=np.roll(lexicon.Chinese, 1)), da, db, lexicon)
        assert np.allclose(same, 1, atol=1e-5)
        assert (other < 0.999).all() and (other >= 0).all()
        assert np.allclose(self_fidelity(da, Distributions(a, role, min_count=1)), 1, atol=1e-5)
