"""The whole pipeline on a small synthetic world."""
import numpy as np
import pandas as pd
import pytest

from twobridges import TwoBridges, WorldConfig, evaluate, make_world, split_of

SMALL = WorldConfig(n_concepts=240, n_topics=8, participants=30)


@pytest.fixture(scope="module")
def world():
    return make_world(SMALL, seed=3)


@pytest.fixture(scope="module")
def model(world):
    return TwoBridges(world.associations("en"), world.associations("zh"), world.lexicon, rank=16, seed=3)


def test_world_is_reproducible(world):
    again = make_world(SMALL, seed=3)
    pd.testing.assert_frame_equal(world.lexicon, again.lexicon)
    pd.testing.assert_frame_equal(world.associations("zh", 10), again.associations("zh", 10))
    assert not make_world(SMALL, seed=4).lexicon.truth.equals(world.lexicon.truth)


def test_participants_give_three_distinct_responses_and_never_the_cue(world):
    long = world.associations("en")
    assert (long.cue != long.response).all()
    per_trial = long.groupby(["pid", "cue"]).response.agg(["size", "nunique"])
    assert (per_trial["size"] == 3).all() and (per_trial["nunique"] == 3).all()
    assert world.associations("en", 10).pid.nunique() < long.pid.nunique()


def test_split_is_fixed_and_roughly_even(world):
    assert [split_of(w) for w in ("en_0000", "en_0001", "moon")] == [split_of(w) for w in ("en_0000", "en_0001", "moon")]
    share = (world.lexicon.English.drop_duplicates().map(split_of) == "dev").mean()
    assert 0.4 < share < 0.6


def test_scores_recover_the_truth(world, model):
    scores = model.score().merge(world.lexicon, on=["English", "Chinese"])
    assert scores.a_star.notna().all()                       # every dictionary pair is covered by one role or the other
    assert scores.rc_baseline.isna().any()                   # the baseline needs both words to have been cues
    result = evaluate(scores, ["rc_baseline", "a_star"], "truth", n_boot=50).set_index("score")
    assert result.rho["a_star"] > 0.3
    assert result.lo["a_star"] <= result.rho["a_star"] <= result.hi["a_star"]


def test_scoring_given_pairs_uses_the_frozen_constants(world, model):
    head = world.lexicon.head(25)[["English", "Chinese"]]
    cols = ["rc_u", "rc_v", "f_fwd", "f_bwd", "a_star"]
    assert np.allclose(model.score(head)[cols], model.score().head(25)[cols], equal_nan=True)
    probe = world.probe_pairs(100, seed=1)
    assert not set(zip(probe.English, probe.Chinese)) & set(zip(world.lexicon.English, world.lexicon.Chinese))
    assert model.score(probe).a_star.notna().all()
