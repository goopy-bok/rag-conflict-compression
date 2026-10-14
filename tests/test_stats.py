import math

import pytest
from statsmodels.stats.multitest import multipletests

from stats import (binom_p, cohen_kappa, cohens_h, fisher_p, holm, mcnemar_exact_test, mcnemar_p,
                   paired_bootstrap, required_n_mcnemar, seed_mean_per_instance)


def test_h3_exact_binomial_not_observed_bootstrap_tail():
    # Amendment 2.2-i: 0/20 successes under H0 p = 0.1 has p = 0.9^20 = 0.1216 (the old observed-sample
    # bootstrap tail would give 0 here); the G1-lite H3 figure 4/200 under p = 0.9 is astronomically small.
    assert binom_p(0, 20, 0.1) == pytest.approx(0.9 ** 20)
    assert binom_p(4, 200, 0.9) < 1e-100
    assert binom_p(180, 200, 0.9) == pytest.approx(0.5, abs=0.1)     # at the null, p is not small


def test_fisher_p_matches_audit_values():
    assert fisher_p(4, 200, 5, 100) == pytest.approx(0.165835, abs=1e-5)
    assert fisher_p(4, 200, 12, 154) == pytest.approx(0.017377, abs=1e-5)
    assert fisher_p(10, 100, 10, 100) == pytest.approx(1.0)


def test_mcnemar_exact_b1_c7():
    # 8 discordant pairs, two-sided exact binomial: 2 * (C(8,0) + C(8,1)) / 2^8 = 18/256
    a = [1] * 1 + [0] * 7 + [1] * 3 + [0] * 39     # C0
    b = [0] * 1 + [1] * 7 + [1] * 3 + [0] * 39     # C1
    assert mcnemar_p(a, b) == pytest.approx(18 / 256)
    assert mcnemar_exact_test(1, 7) == pytest.approx(18 / 256)   # legacy function agrees


def test_holm_order_and_values():
    p = [0.01, 0.04, 0.03, 0.005]
    assert holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm(p) == pytest.approx(list(multipletests(p, method="holm")[1]))


def test_cohen_kappa_textbook():
    # 50 items: yes/yes 20, yes/no 5, no/yes 10, no/no 15 -> po 0.7, pe 0.5, kappa 0.4
    x = ["y"] * 25 + ["n"] * 25
    y = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(x, y) == pytest.approx(0.4)
    assert cohen_kappa([1, 0, 1], [1, 0, 1]) == pytest.approx(1.0)


def test_cohens_h():
    assert cohens_h(0.5, 0.5) == 0
    assert cohens_h(1.0, 0.0) == pytest.approx(math.pi)
    assert cohens_h(0.14, 0.02) == pytest.approx(2 * math.asin(math.sqrt(0.14)) - 2 * math.asin(math.sqrt(0.02)))


def test_paired_bootstrap_reproducible():
    a = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1]
    b = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    r1, r2 = paired_bootstrap(a, b, n=2000, seed=3), paired_bootstrap(a, b, n=2000, seed=3)
    assert r1 == r2
    assert r1[0] == pytest.approx(0.4)
    assert r1[1][0] <= 0.4 <= r1[1][1]


def test_required_n():
    # (1.959964 + 0.841621)^2 * 0.16 / 0.05^2 = 502.3 -> 503
    assert required_n_mcnemar(0.16) == 503
    assert required_n_mcnemar(0.16, delta=0.1) == 126


def test_seed_mean_per_instance():
    by_seed = {0: {"a": {"csr": 1}, "b": {"csr": 0}}, 1: {"a": {"csr": 1}, "b": {"csr": 1}},
               2: {"a": {"csr": 0}, "b": {"csr": 1}, "c": {"csr": 1}}}
    assert seed_mean_per_instance(by_seed) == pytest.approx({"a": 2 / 3, "b": 2 / 3})
