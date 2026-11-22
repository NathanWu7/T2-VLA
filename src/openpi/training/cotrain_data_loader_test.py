from types import SimpleNamespace

import numpy as np
import pytest

from openpi.training.data_loader import (
    MixtureDataset,
    StratifiedBatchSampler,
    _frame_indices_for_episodes,
    validate_dataset_compatibility,
)
from openpi.shared.normalize import RunningStats


class _SyntheticDataset:
    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return {"value": np.asarray(self.values[index], dtype=np.float32)}


def test_frame_indices_for_interleaved_episode_subset():
    episode_indices = np.asarray([0, 1, 0, 2, 1, 3, 2, 1, 3])
    np.testing.assert_array_equal(
        _frame_indices_for_episodes(episode_indices, (1, 3)),
        np.asarray([1, 4, 5, 7, 8]),
    )


def test_stratified_sampler_exact_fraction_and_determinism():
    sampler = StratifiedBatchSampler(7, 5, batch_size=10, sim_weight=0.3, seed=17, num_batches=25)
    batches = list(sampler)
    assert all(sum(index >= 7 for index in batch) == 3 for batch in batches)

    same = StratifiedBatchSampler(7, 5, batch_size=10, sim_weight=0.3, seed=17, num_batches=25)
    assert batches == list(same)


@pytest.mark.parametrize(("sim_weight", "expected_sim"), [(0.0, 0), (1.0, 8)])
def test_stratified_sampler_degenerate_weights(sim_weight, expected_sim):
    sampler = StratifiedBatchSampler(4, 4, batch_size=8, sim_weight=sim_weight, seed=0, num_batches=3)
    assert all(sum(index >= 4 for index in batch) == expected_sim for batch in sampler)


def test_stratified_sampler_rejects_non_integer_count():
    with pytest.raises(ValueError, match="must be an integer"):
        StratifiedBatchSampler(4, 4, batch_size=7, sim_weight=0.5)


def test_stratified_sampler_resume_reproduces_subsequent_batches():
    sampler = StratifiedBatchSampler(9, 3, batch_size=6, sim_weight=0.5, seed=4, num_batches=10)
    iterator = iter(sampler)
    for _ in range(3):
        next(iterator)
    state = sampler.state_dict()
    assert state["batches_consumed"] == 3
    expected = [next(iterator) for _ in range(3)]

    resumed = StratifiedBatchSampler(9, 3, batch_size=6, sim_weight=0.5, seed=999, num_batches=10)
    resumed.load_state_dict(state)
    resumed_iterator = iter(resumed)
    assert [next(resumed_iterator) for _ in range(3)] == expected


def test_mixture_dataset_indexing_and_domain():
    dataset = MixtureDataset(_SyntheticDataset([1, 2]), _SyntheticDataset([3]))
    assert len(dataset) == 3
    assert dataset[0]["domain"] == 0
    assert dataset[-1]["domain"] == 1
    assert dataset[2]["value"] == 3


def test_incompatible_dataset_metadata_raises():
    real = SimpleNamespace(fps=20, features={"state": {"shape": [3]}, "actions": {"shape": [4]}})
    sim = SimpleNamespace(fps=20, features={"state": {"shape": [3]}, "actions": {"shape": [5]}})
    with pytest.raises(ValueError, match="incompatible shapes"):
        validate_dataset_compatibility(real, sim)


def test_union_stats_are_sample_count_weighted():
    real = np.asarray([[0.0, 1.0], [2.0, 3.0]])
    sim = np.asarray([[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]])
    union = np.concatenate((real, sim), axis=0)
    cotrain_stats = RunningStats()
    plain_stats = RunningStats()
    for start in range(0, len(union), 2):
        cotrain_stats.update(union[start : start + 2])
        plain_stats.update(union[start : start + 2])
    got, expected = cotrain_stats.get_statistics(), plain_stats.get_statistics()
    np.testing.assert_allclose(got.mean, expected.mean)
    np.testing.assert_allclose(got.mean, union.mean(axis=0))
    np.testing.assert_allclose(got.std, expected.std)
    np.testing.assert_allclose(got.q01, expected.q01)
    np.testing.assert_allclose(got.q99, expected.q99)
