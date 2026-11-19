"""Tests for hydra_tod.map_sampler.sample_map_posterior and the extensions
of linear_sampler it builds on (diagonal and additive noise in
params_space_oper_and_data; rng and no input mutation in sample_p_v2).

The oracle builds the posterior precision with explicit diagonal matrices,
independently of the code under test:

    P = sum_i U_i^T D_i^-1 N_i^-1 D_i^-1 U_i + S^-1,   D_i = diag(U_i p_hat + mu_i)

(D_i = I under the additive model). With L L^T = P, the whitened draws
(x - p_hat) L must be N(0, I).
"""

import tracemalloc

import numpy as np
import pytest

from hydra_tod.linear_sampler import params_space_oper_and_data, sample_p_v2
from hydra_tod.map_sampler import sample_map_posterior

N_PAR = 12
LENGTHS = (150, 110)
FRAC = 1e-3  # fractional noise rms
N_DRAWS = 4000


def _correlated_cov(n, corr_len=5.0, white=0.3):
    """A positive-definite, 1/f-like (correlated) covariance."""
    lag = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    return np.exp(-lag / corr_len) + white * np.eye(n)


@pytest.fixture(scope="module")
def system():
    rng = np.random.default_rng(1)
    U = [rng.random((n, N_PAR)) + 0.1 for n in LENGTHS]
    p_true = rng.random(N_PAR) * 10.0 + 2.0
    mu = [np.full(n, 20.0) for n in LENGTHS]
    N = [FRAC**2 * _correlated_cov(n) for n in LENGTHS]
    d = [
        (Ui @ p_true + m) * (1.0 + rng.multivariate_normal(np.zeros(n), Ni))
        for Ui, m, Ni, n in zip(U, mu, N, LENGTHS)
    ]
    Ninv = [np.linalg.inv(Ni) for Ni in N]
    w = [np.full(n, 1.0 / FRAC**2) * rng.uniform(0.5, 2.0, n) for n in LENGTHS]
    # any centre: the sampler is centred on the p_hat it is given
    p_hat = p_true + rng.normal(0.0, 1e-2, N_PAR)
    return U, d, mu, Ninv, w, p_hat


def _oracle_precision(U, Ninv, mu, p_hat, multiplicative=True, prior=None):
    A = np.zeros((N_PAR, N_PAR))
    for Ui, Ni, m in zip(U, Ninv, mu):
        n = Ui.shape[0]
        Dinv = np.diag(1.0 / (Ui @ p_hat + m)) if multiplicative else np.eye(n)
        Nmat = np.diag(Ni) if np.ndim(Ni) == 1 else Ni
        A += Ui.T @ Dinv @ Nmat @ Dinv @ Ui
    if prior is not None:
        A += np.diag(prior) if prior.ndim == 1 else prior
    return A


def _assert_draws_follow(draws, p_hat, precision):
    """Whitened by chol(precision), the draws are N(0, I) around p_hat."""
    W = (draws - p_hat) @ np.linalg.cholesky(precision)
    n, k = W.shape
    assert abs((W**2).sum(axis=1).mean() / k - 1.0) < 0.03
    assert np.abs(W.mean(axis=0)).max() * np.sqrt(n) < 4.5
    C = np.cov(W.T) - np.eye(k)
    assert np.abs(C).max() < 5.0 * np.sqrt(2.0 / n)


def _same_up_to_roundoff(x, y, precision):
    bound = 10.0 * np.finfo(float).eps * np.linalg.cond(precision) * np.abs(y).max()
    np.testing.assert_allclose(x, y, rtol=0.0, atol=bound)


# ---------------------------------------------------------------------- #
# params_space_oper_and_data: diagonal and additive noise                #
# ---------------------------------------------------------------------- #
class TestParamsSpaceOperAndData:
    @pytest.mark.parametrize("multiplicative", [True, False])
    def test_diagonal_equals_dense(self, system, multiplicative):
        U, d, mu, Ninv, w, p_hat = system
        wi = w[0].copy()
        wi[10:20] = 0.0  # flagged samples
        s = np.sqrt(wi)
        A1, b1, T1 = params_space_oper_and_data(
            d[0], U[0], p_hat, wi, mu=mu[0], Ninv_sqrt=s, multiplicative=multiplicative
        )
        A2, b2, T2 = params_space_oper_and_data(
            d[0], U[0], p_hat, np.diag(wi), mu=mu[0], Ninv_sqrt=np.diag(s),
            multiplicative=multiplicative,
        )
        np.testing.assert_allclose(A1, A2, rtol=1e-12)
        np.testing.assert_allclose(b1, b2, rtol=1e-12)
        np.testing.assert_allclose(T1, T2, rtol=1e-12)

    @pytest.mark.parametrize("multiplicative", [True, False])
    def test_matches_explicit_formula(self, system, multiplicative):
        U, d, mu, Ninv, w, p_hat = system
        A, b = params_space_oper_and_data(
            d[1], U[1], p_hat, Ninv[1], mu=mu[1], multiplicative=multiplicative
        )
        n = U[1].shape[0]
        Dinv = np.diag(1.0 / (U[1] @ p_hat + mu[1])) if multiplicative else np.eye(n)
        S = Dinv @ Ninv[1] @ Dinv
        np.testing.assert_allclose(A, U[1].T @ S @ U[1], rtol=1e-10)
        np.testing.assert_allclose(b, U[1].T @ S @ (d[1] - mu[1]), rtol=1e-10)


# ---------------------------------------------------------------------- #
# sample_p_v2: rng and inputs left unchanged                              #
# ---------------------------------------------------------------------- #
class TestSamplePV2:
    @pytest.mark.parametrize("prior_kind", ["1d", "2d"])
    @pytest.mark.parametrize("est_mode", [True, False])
    def test_inputs_not_mutated(self, prior_kind, est_mode):
        rng = np.random.default_rng(2)
        M = rng.random((5, 5))
        A = M @ M.T + 5 * np.eye(5)
        b, wn, m = rng.random(5), rng.random(5), rng.random(5)
        prior = np.full(5, 2.0) if prior_kind == "1d" else 2.0 * np.eye(5)
        A0, b0, prior0 = A.copy(), b.copy(), prior.copy()
        sample_p_v2(A, b, wn, prior_cov_inv=prior, prior_mean=m, solver=None,
                    Est_mode=est_mode, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(A, A0)
        np.testing.assert_array_equal(b, b0)
        np.testing.assert_array_equal(prior, prior0)

    def test_rng_reproducible_and_global_default(self):
        A, b, wn, m = 3.0 * np.eye(4), np.ones(4), np.zeros(4), np.zeros(4)
        kw = dict(prior_cov_inv=np.ones(4), prior_mean=m, solver=None, Est_mode=False)
        x1 = sample_p_v2(A, b, wn, rng=np.random.default_rng(7), **kw)
        x2 = sample_p_v2(A, b, wn, rng=np.random.default_rng(7), **kw)
        np.testing.assert_array_equal(x1, x2)
        np.random.seed(3)
        y1 = sample_p_v2(A, b, wn, **kw)
        np.random.seed(3)
        y2 = sample_p_v2(A, b, wn, **kw)
        np.testing.assert_array_equal(y1, y2)

    def test_white_noise_term_required_only_for_sampling(self):
        A, b = 3.0 * np.eye(4), np.ones(4)
        estimate = sample_p_v2(A, b, None, solver=None, Est_mode=True)
        np.testing.assert_allclose(estimate, b / 3.0, rtol=1e-14)
        with pytest.raises(ValueError, match="A_sqrt_wn is required"):
            sample_p_v2(A, b, None, solver=None, Est_mode=False)


# ---------------------------------------------------------------------- #
# sample_map_posterior: distribution                                      #
# ---------------------------------------------------------------------- #
class TestDistribution:
    def test_multiplicative_correlated_noise_diag_prior(self, system):
        U, d, mu, Ninv, w, p_hat = system
        prior = np.full(N_PAR, 1e3)
        draws, precision = sample_map_posterior(
            d, U, Ninv, p_hat, mu_list=mu, prior_cov_inv=prior,
            n_samples=N_DRAWS, rng=1,
        )
        oracle = _oracle_precision(U, Ninv, mu, p_hat, prior=prior)
        np.testing.assert_allclose(precision, oracle, rtol=1e-9)
        _assert_draws_follow(draws, p_hat, oracle)

    def test_additive_no_prior(self, system):
        U, d, mu, Ninv, w, p_hat = system
        draws, precision = sample_map_posterior(
            d, U, Ninv, p_hat, mu_list=mu, multiplicative=False,
            n_samples=N_DRAWS, rng=2,
        )
        oracle = _oracle_precision(U, Ninv, mu, p_hat, multiplicative=False)
        np.testing.assert_allclose(precision, oracle, rtol=1e-9)
        _assert_draws_follow(draws, p_hat, oracle)

    def test_diagonal_flagged_full_prior(self, system):
        U, d, mu, Ninv, w, p_hat = system
        wf = [x.copy() for x in w]
        wf[0][5:25] = 0.0
        L = np.random.default_rng(3).random((N_PAR, N_PAR))
        prior = 1e2 * (L @ L.T + N_PAR * np.eye(N_PAR))
        draws, precision = sample_map_posterior(
            d, U, wf, p_hat, mu_list=mu, prior_cov_inv=prior,
            n_samples=N_DRAWS, rng=3,
        )
        oracle = _oracle_precision(U, wf, mu, p_hat, prior=prior)
        np.testing.assert_allclose(precision, oracle, rtol=1e-9)
        _assert_draws_follow(draws, p_hat, oracle)

    def test_semidefinite_dense_noise(self, system):
        """Flagged samples zeroed out of a dense N^-1."""
        U, d, mu, Ninv, w, p_hat = system
        Nf = [x.copy() for x in Ninv]
        Nf[1][:, 30:40] = 0.0
        Nf[1][30:40, :] = 0.0
        draws, precision = sample_map_posterior(
            d, U, Nf, p_hat, mu_list=mu, n_samples=N_DRAWS, rng=4
        )
        oracle = _oracle_precision(U, Nf, mu, p_hat)
        _assert_draws_follow(draws, p_hat, oracle)

    def test_fully_flagged_diagonal_tod(self, system):
        U, d, mu, Ninv, w, p_hat = system
        draws, precision = sample_map_posterior(
            d, U, [w[0], np.zeros(LENGTHS[1])], p_hat, mu_list=mu,
            prior_cov_inv=np.full(N_PAR, 1.0),
            n_samples=N_DRAWS, rng=5,
        )
        oracle = _oracle_precision(U, [w[0], np.zeros(LENGTHS[1])], mu, p_hat,
                                   prior=np.full(N_PAR, 1.0))
        _assert_draws_follow(draws, p_hat, oracle)

    @pytest.mark.parametrize("multiplicative", [True, False])
    def test_diagonal_and_dense_give_the_same_draws(self, system, multiplicative):
        """Boundary check: the same white noise stored two ways. With all
        weights positive the dense square root is chol(diag(w)) =
        diag(sqrt(w)), so one seed gives the same draws."""
        U, d, mu, Ninv, w, p_hat = system
        kw = dict(mu_list=mu, prior_cov_inv=np.full(N_PAR, 10.0),
                  multiplicative=multiplicative, n_samples=30, rng=6)
        x1, P1 = sample_map_posterior(d, U, w, p_hat, **kw)
        x2, P2 = sample_map_posterior(d, U, [np.diag(x) for x in w], p_hat, **kw)
        _same_up_to_roundoff(x1, x2, P2)


# ---------------------------------------------------------------------- #
# sample_map_posterior: rng                                               #
# ---------------------------------------------------------------------- #
class TestRng:
    def test_int_seed_reproducible_and_global_state_untouched(self, system):
        U, d, mu, Ninv, w, p_hat = system
        np.random.seed(11)
        before = np.random.get_state()[1].copy()
        a, _ = sample_map_posterior(d, U, w, p_hat, mu_list=mu, n_samples=3, rng=9)
        b, _ = sample_map_posterior(d, U, w, p_hat, mu_list=mu, n_samples=3, rng=9)
        c, _ = sample_map_posterior(d, U, w, p_hat, mu_list=mu, n_samples=3, rng=10)
        np.testing.assert_array_equal(np.random.get_state()[1], before)
        np.testing.assert_array_equal(a, b)
        assert not np.allclose(a, c)

    def test_generator(self, system):
        U, d, mu, Ninv, w, p_hat = system
        a, _ = sample_map_posterior(d, U, w, p_hat, mu_list=mu, n_samples=3,
                                    rng=np.random.default_rng(4))
        b, _ = sample_map_posterior(d, U, w, p_hat, mu_list=mu, n_samples=3,
                                    rng=np.random.default_rng(4))
        np.testing.assert_array_equal(a, b)

    def test_none_uses_global_stream(self, system):
        U, d, mu, Ninv, w, p_hat = system
        np.random.seed(5)
        a, _ = sample_map_posterior(d, U, w, p_hat, mu_list=mu, n_samples=3)
        np.random.seed(5)
        b, _ = sample_map_posterior(d, U, w, p_hat, mu_list=mu, n_samples=3)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------- #
# sample_map_posterior: validation                                        #
# ---------------------------------------------------------------------- #
class TestValidation:
    def test_length_mismatch(self, system):
        U, d, mu, Ninv, w, p_hat = system
        with pytest.raises(ValueError, match="same length"):
            sample_map_posterior(d, U, w[:1], p_hat)

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True])
    def test_n_samples(self, system, bad):
        U, d, mu, Ninv, w, p_hat = system
        with pytest.raises(ValueError, match="n_samples"):
            sample_map_posterior(d, U, w, p_hat, n_samples=bad)

    def test_noise_inv_shape(self, system):
        U, d, mu, Ninv, w, p_hat = system
        with pytest.raises(ValueError, match=r"Ninv_list\[1\]"):
            sample_map_posterior(d, U, [w[0], w[1][:-1]], p_hat)

    def test_negative_diagonal(self, system):
        U, d, mu, Ninv, w, p_hat = system
        bad = w[1].copy()
        bad[0] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            sample_map_posterior(d, U, [w[0], bad], p_hat)

    def test_non_finite_diagonal(self, system):
        U, d, mu, Ninv, w, p_hat = system
        bad = w[1].copy()
        bad[0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            sample_map_posterior(d, U, [w[0], bad], p_hat)

    def test_indefinite_dense(self, system):
        U, d, mu, Ninv, w, p_hat = system
        bad = np.eye(LENGTHS[1])
        bad[0, 0] = -1.0
        with pytest.raises(ValueError, match="semi-definite"):
            sample_map_posterior(d, U, [Ninv[0], bad], p_hat)

    def test_singular_full_prior(self, system):
        U, d, mu, Ninv, w, p_hat = system
        prior = np.zeros((N_PAR, N_PAR))
        prior[:2, :2] = [[1.0, 0.2], [0.2, 1.0]]
        with pytest.raises(ValueError, match="positive definite"):
            sample_map_posterior(d, U, w, p_hat, prior_cov_inv=prior)

    def test_improper_posterior(self, system):
        """A direction constrained by neither the data nor the prior: two
        identical operator columns, no prior on either."""
        U, d, mu, Ninv, w, p_hat = system
        U2 = [Ui.copy() for Ui in U]
        for Ui in U2:
            Ui[:, 1] = Ui[:, 0]
        prior = np.full(N_PAR, 1.0)
        prior[:2] = 0.0
        with pytest.raises(np.linalg.LinAlgError, match="neither the data nor the prior"):
            sample_map_posterior(d, U2, w, p_hat, mu_list=mu, prior_cov_inv=prior)

    def test_asymmetric_noise_inv(self, system):
        U, d, mu, Ninv, w, p_hat = system
        bad = Ninv[1].copy()
        bad[0, 1] += 1e-3 * np.abs(bad).max()
        with pytest.raises(ValueError, match="symmetric"):
            sample_map_posterior(d, U, [Ninv[0], bad], p_hat, mu_list=mu)

    def test_asymmetric_prior(self, system):
        U, d, mu, Ninv, w, p_hat = system
        prior = np.eye(N_PAR)
        prior[0, 1] = 0.5
        with pytest.raises(ValueError, match="symmetric"):
            sample_map_posterior(d, U, w, p_hat, prior_cov_inv=prior)

    def test_model_through_zero(self, system):
        U, d, mu, Ninv, w, p_hat = system
        with pytest.raises(FloatingPointError, match="zero"):
            sample_map_posterior(d, U, w, np.zeros(N_PAR), mu_list=[0.0, 0.0])


# ---------------------------------------------------------------------- #
# Scaling: memory of long diagonal-noise TODs                            #
# ---------------------------------------------------------------------- #
def test_ill_conditioned_narrow_posterior_keeps_its_spread():
    """cond(P) ~ 8e10 and posterior sd ~ 2e-12 |p_hat|. Solving
    P x = P p_hat + noise loses the spread to the rounding of P p_hat,
    amplified by cond(P) (whitened sd of 23 and 90 here); drawing the
    zero-mean fluctuation and adding p_hat keeps it."""
    n = 400
    t = np.linspace(0.0, 1.0, n)
    U = np.column_stack([np.ones(n), np.ones(n) + 1e-4 * t, t**2])
    p = np.array([10.0, 20.0, 30.0])
    draws, precision = sample_map_posterior(
        [U @ p], [U], [np.full(n, 1e20)], p,
        multiplicative=False, n_samples=3000, rng=1,
    )
    W = (draws - p) @ np.linalg.cholesky(precision)
    assert np.all(np.abs(W.std(axis=0) - 1.0) < 0.1), W.std(axis=0)


def test_long_diagonal_tods_stay_linear_in_memory():
    n, k = 8000, 10  # a dense N^-1 would be 512 MB per TOD
    rng = np.random.default_rng(8)
    U = [rng.random((n, k)) + 0.1 for _ in range(2)]
    p = rng.random(k) * 10 + 2
    d = [Ui @ p for Ui in U]
    w = [np.full(n, 1e6) for _ in range(2)]
    tracemalloc.start()
    try:
        draws, _ = sample_map_posterior(d, U, w, p, n_samples=20, rng=0)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < n * n * 8 / 10, f"peak {peak / 1e6:.0f} MB"
    assert draws.shape == (20, k) and np.all(np.isfinite(draws))
