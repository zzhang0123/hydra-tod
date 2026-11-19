"""Tests for the joint system-temperature projection operator."""

from types import SimpleNamespace

import numpy as np
from scipy.linalg import block_diag


def test_get_tsys_operator_places_each_tod_in_its_own_columns(monkeypatch):
    from hydra_tod import full_Gibbs_sampler

    local_operators = [
        np.ones((2, 1)),
        np.ones((3, 1)),
        np.ones((1, 1)),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    ]
    sky_operators = [np.eye(op.shape[0], 3) for op in local_operators]

    comm = SimpleNamespace(allgather=lambda local_size: [2, local_size, 3])
    monkeypatch.setattr(full_Gibbs_sampler, "comm", comm)
    monkeypatch.setattr(full_Gibbs_sampler, "rank", 1)

    actual = full_Gibbs_sampler.get_Tsys_operator(sky_operators, local_operators)
    expected_local = np.pad(block_diag(*local_operators), ((0, 0), (2, 3)))
    expected = np.hstack((np.vstack(sky_operators), expected_local))

    assert len(actual) == len(local_operators)
    assert [op.shape[0] for op in actual] == [op.shape[0] for op in local_operators]
    np.testing.assert_array_equal(np.vstack(actual), expected)
