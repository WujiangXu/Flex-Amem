"""Endpoint statistics (docs/protocol/experiment_protocol.md): exact McNemar, paired bootstrap clustered by repository
(10k resamples, seeded), bootstrap TOST non-inferiority, Wilson intervals, Holm correction, judge agreement."""
import math

import numpy as np
from scipy.stats import binom, norm

N_BOOT = 10_000
SEED = 20260924  # the split seed (configs/splits/manifest.json)
MIN_DISCORDANT = 5  # below this, the bootstrap CI of a paired binary difference is degenerate: NI not evaluable


def mcnemar_exact(a, b) -> dict:
    """Exact two-sided McNemar test on paired binary outcomes a, b (same items, same order):
    binomial test of the discordant pairs n10 (a right, b wrong) vs n01 at p = 0.5."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    return {"n": len(a), "n10": n10, "n01": n01,
            "p": 1.0 if n == 0 else min(1.0, 2 * float(binom.cdf(min(n10, n01), n, 0.5)))}


def paired_bootstrap(x, clusters=None, n_boot: int = N_BOOT, seed: int = SEED, alpha: float = 0.05) -> dict:
    """Mean of the per-item paired quantity x (an accuracy difference a_i - b_i, or a log cost ratio) with a
    percentile (1 - alpha) CI. Whole clusters (repositories for SWE) are resampled with replacement and the
    statistic is the mean over all items of the resampled clusters; clusters=None resamples items. `discordant` is
    the number of nonzero x (for an accuracy difference, the discordant pairs)."""
    x = np.asarray(x, float)
    if not len(x):
        return {"n": 0, "k": 0, "discordant": 0, "est": math.nan, "lo": math.nan, "hi": math.nan, "boot": np.array([])}
    ids = np.arange(len(x)) if clusters is None else np.unique(np.asarray(clusters), return_inverse=True)[1]
    sums, cnt = np.bincount(ids, weights=x), np.bincount(ids)
    idx = np.random.default_rng(seed).integers(0, len(sums), size=(n_boot, len(sums)))
    boot = sums[idx].sum(1) / cnt[idx].sum(1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"n": len(x), "k": len(sums), "discordant": int(np.count_nonzero(x)), "est": float(x.mean()),
            "lo": float(lo), "hi": float(hi), "boot": boot}


def noninferiority(a, b, delta: float, clusters=None, alpha: float = 0.05, **kw) -> dict:
    """TOST-style non-inferiority of a against b on paired binary outcomes, margin delta (a proportion, e.g.
    0.03). H0: p_a - p_b <= -delta. The two one-sided tests at level alpha are read off the (1 - 2 alpha)
    percentile CI of the paired bootstrap of the item differences, clustered by repository for SWE (the protocol
    names the clustered bootstrap as the SWE test and no other interval; LongMemEval passes clusters=None).
    a is non-inferior if the lower limit exceeds -delta, equivalent if also the upper limit is below delta;
    p is the bootstrap one-sided p-value P*(diff <= -delta). Not evaluable (never non-inferior) with fewer than
    MIN_DISCORDANT discordant pairs: with none, the CI is [0, 0] and would pass any margin."""
    d = np.asarray(a, float) - np.asarray(b, float)
    bs = paired_bootstrap(d, clusters, alpha=2 * alpha, **kw)
    ev = bs["discordant"] >= MIN_DISCORDANT
    ok = ev and bs["lo"] > -delta
    return {**{k: v for k, v in bs.items() if k != "boot"}, "delta": delta, "evaluable": ev, "noninferior": ok,
            "equivalent": ok and bs["hi"] < delta, "p": float((bs["boot"] <= -delta).mean()) if bs["n"] else math.nan}


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """(p, lo, hi): proportion with its Wilson score interval."""
    if n == 0:
        return math.nan, math.nan, math.nan
    z, p = norm.ppf(1 - alpha / 2), k / n
    den = 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / den, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (same order as the input)."""
    p = np.asarray(pvals, float)
    o = np.argsort(p)
    adj = np.minimum(1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]))
    out = np.empty(len(p))
    out[o] = adj
    return out


def agreement(a, b) -> tuple[float, float]:
    """(raw agreement, Cohen's kappa) of two binary label vectors."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if not len(a):
        return math.nan, math.nan
    po = float((a == b).mean())
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return po, (po - pe) / (1 - pe) if pe < 1 else math.nan
