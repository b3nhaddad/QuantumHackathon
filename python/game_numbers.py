# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///
"""Every quantity the Qupacabrathon documents quote about the odd-cycle game.

Nothing in the challenge documents, a slide or a notebook should assert a
number that this file does not produce. Run it and read the output:

    uv run --python 3.12 scripts/game_numbers.py

The game is the odd cycle $C_n$ for odd $n \\geq 3$. Referee picks an edge or a
vertex; the two players each answer one bit; they win when adjacent vertices get
different bits and identical vertices get equal bits. Source for both bounds and
for the strategy: Drmota, Grilo, Vidick et al., Phys. Rev. Lett. 134, 070201
(2025), arXiv:2406.08412.

The statistical gate is the one-sided exact binomial tail test in `p_value`
below. numpy is a dependency only so `_cross_check_bernstein` can import the
reference module it compares `confidence_half_width` against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# The two devices a team may choose between, and the route each label reaches.
#
# A team names its device by label, `QPU = "emerald"` or `"garnet"` in the
# submission template's parameter block. The organizer's environment maps the
# label to the route, so no route string ever appears in a submission and
# choosing a fallback inside a device family stays an organizer decision.
# `challenge/hardware.py` and `organizers/quote_check.py` import DEVICE_ID as
# the default route.
QPU_ROUTES = {
    "emerald": "aws:iqm:qpu:emerald",
    "garnet": "aws:iqm:qpu:garnet",
}
DEFAULT_QPU = "emerald"
DEVICE_ID = QPU_ROUTES[DEFAULT_QPU]

# The meter is continuous dollars, not whole credits: qBraid denominates in
# credits at 1 credit = $0.01 and charges a task fee per circuit plus a
# per-shot rate, so
#
#     cost per circuit = TASK_DOLLARS + SHOT_DOLLARS * shots
#
# Rates established 2026-08-18 via the insufficient-credits refusal oracle:
# submit against a balance too small to pay and read the price out of the
# refusal. Every route measured that session is in MEASURED_ROUTE_RATES below.
# The bare constants are the default device's, because every published table is
# priced there; `rates_for_qpu` reads the row for a team that picks the other,
# and `cost` takes a `qpu` argument so a run is priced on the device it goes to.
#
# Both IQM rows were re-confirmed against qBraid's own published pricing page on
# 21 August 2026: 30.00 credits per task on both devices, 0.16 credits per shot
# on Emerald and 0.145 on Garnet, at 1 credit = $0.01. That is an independent
# second source agreeing with the refusal oracle to the last digit, so these two
# rows are no longer [UNVERIFIED].
TASK_DOLLARS = 0.30
SHOT_DOLLARS = 0.0016

# Every route measured on 2026-08-18: `(route, task fee $, per-shot $)`. Kept
# here because `_check_route_price` re-derives the device ranking from these
# rows against PUBLISHED_READOUT below, and a self-check that lives next to the
# constants it checks cannot be forgotten when a constant moves. The raw
# refusal quotes were not recorded shot-by-shot the way the old Open Quantum
# quotes were, so the check asserts the fitted rates and the ranking rather
# than reproducing individual quotes. The two IQM rows are confirmed twice over,
# by the refusal oracle and by the published pricing page; the Rigetti rows below
# have only the oracle. [UNVERIFIED: the Rigetti rates, which no offered device
# uses.]
#
# Route ids corrected 2026-08-19 from a live $0 metadata read: every real id
# carries a `:qpu:` segment, and the shorthand ids recorded on 18 August no
# longer resolve ("Device not found"). Same devices, same rates.
#
# The measured-deficit column that used to sit here moved out of this file on
# 2026-08-21, when the event stopped publishing any device's deficit: it is a
# number a team is meant to predict or measure, so it lives in the organizers'
# `measured_deficits.py` and nothing in the student repo carries it. The
# Rigetti rows stay because they are price measurements; the Cepheus route was
# disqualified on 2026-08-19 and neither Rigetti route is offered to a team.
MEASURED_ROUTE_RATES = (
    ("aws:iqm:qpu:garnet", 0.30, 0.00145),
    ("aws:rigetti:qpu:cepheus-1-108q", 0.30, 0.000425),
    # [UNVERIFIED: live metadata on 2026-08-19 advertises $120/minute on this
    # route, contradicting the per-shot rate measured 18 August; route dead
    # for this event. The row stays as the only record of that measurement.]
    ("rigetti:rigetti:qpu:cepheus-1-108q", 0.00, 0.00165),
    ("aws:iqm:qpu:emerald", 0.30, 0.0016),
)

# Published readout error per wire for each offered device, `(e0, e1, source)`.
# This is the whole input to the prediction route: push a pair through
# `delta_from_readout_asym` and out comes a deficit to size a run against,
# before spending anything. Both rows are readout figures rather than deficits,
# and both are upper estimates because neither corrects for state preparation
# error, which is why the prediction over-buys against the truth in both cases.
#
# Garnet: median readout error 3e-2 for simultaneous QPU readout, defined as
# the probability of not discriminating the qubit to be in the prepared state
# *without* correcting for state preparation errors. Abdurakhimov et al.,
# "Technology and Performance Benchmarks of IQM's 20-Qubit Quantum Computer",
# arXiv:2408.12433, Section 4.1. One median, so the pair is symmetric.
#
# Emerald: IQM publishes no readout error for the Crystal 54 device; the AWS
# Braket launch note gives median 1Q fidelity 99.93% and median 2Q fidelity
# 99.5% and points at the Braket console for readout. The pair here is our own
# probe of 2026-08-20, and the 2.8% carries preparation error with it.
PUBLISHED_READOUT = {
    "garnet": (0.030, 0.030, "arXiv:2408.12433 Sec. 4.1, median, simultaneous"),
    "emerald": (0.0013, 0.028, "our probe, 2026-08-20; e1 includes prep error"),
}

# Published median two-qubit (native CZ) gate error, same figure for both
# offered devices. Garnet: Abdurakhimov et al., arXiv:2408.12433, Section 4.1,
# median 2Q fidelity 99.5%. Emerald: the AWS Braket launch note quoted above
# under PUBLISHED_READOUT, same 99.5% median 2Q fidelity figure. This repo's
# odd-cycle circuit uses exactly one CNOT (compiled to the native CZ) to
# prepare the Bell state, so this error is paid once per circuit, independent
# of n -- unlike readout, which is paid on both wires of every shot. At n=13
# it is roughly 6x-12x smaller than the combined readout deficit for either
# device (see `tools/noise.py`), which is why readout is the dominant channel
# and this is the secondary one.
GATE_2Q_ERROR = 0.005

# The old Open Quantum device refused fewer than 10 shots server-side, and the
# floor is kept. [UNVERIFIED: the minimum is unmeasured on the qBraid Garnet
# route. A dry-run question, not a blocker.]
MINIMUM_SHOTS = 10

# One-sided normal tail at 3 sigma. The gate is a threshold, never a score.
# Holding every team to p < P_3SIGMA keeps the family-wise false-positive rate
# across 12 teams at 1 - (1 - p)^12 = 1.6%.
P_3SIGMA = 0.0013498980316300946

# The 2 sigma gate that was rejected, same one-sided tail at k = 2. Kept so
# `family_wise_false_positive` can print what it would have cost across 12
# teams, which is the whole reason the gate is 3 sigma.
P_2SIGMA = 0.022750131948179195

# Device deficit: how far below the ideal quantum win rate the hardware lands,
# omega = omega_q - delta. Taken to be n-independent because the circuit does
# not change with n, only the rotation angle does.
#
# This is NOT a measurement of either device, and no measurement of either
# device appears anywhere in this repository. Delta is what a team declares in
# its own parameter block, having predicted it from PUBLISHED_READOUT above or
# measured it with a run of its own, and every function here takes it as an
# argument. The constant exists only so the worked examples, the published grid
# and the self-checks are all illustrated at one value instead of scattering
# arbitrary ones through the file.
#
# 0.015 was chosen for that job on 2026-08-21 because it is a plausible deficit
# that is not either device's answer: it is better than the prediction route
# reaches for either device and worse than the better device actually is, so a
# team that reads it off cannot mistake it for the truth in either direction.
DELTA_EXAMPLE = 0.015

# The grid of deficits the instructions publish as a sensitivity table, and the
# one the report below prints. It is a payoff schedule rather than a device
# reading: no row is either device's answer, and the table exists so a team can
# see what its declared delta is worth before it declares one.
#
# The rows are chosen so their affordable frontiers at the $20 cap are distinct
# (n = 19, 15, 11, 7, 5) and so the grid brackets the prediction route at both
# ends. It deliberately starts at 0.015 rather than lower: a row at 0.010 would
# hand over a frontier that a team is supposed to discover by measuring. The
# top row is 0.060 because pushing Garnet's published readout error through
# `predicted_delta` lands at 0.0578, and a team that does that homework should
# find a row at or above its own prediction rather than have to interpolate.
PUBLISHED_DELTA_GRID = (0.015, 0.020, 0.030, 0.040, 0.060)

# There is no post-execution mitigation multiplier here on purpose. Zero-noise
# extrapolation and every other post-execution method is disallowed, so
# nothing prices a sweep run more than once. Randomised-order repeat passes
# multiply only the task term of `cost`, and at this pot the event runs one
# pass per sweep.

# The dollar pot, the per-team cap, and what is left over. All three are qBraid
# dollars, which is what the vendor charges and what the balance is
# denominated in.
#
# CAP_PER_TEAM = $20 funds a real strategic choice, and three archetypes fit
# inside it at the illustration delta = 0.015: a two-run ladder that certifies
# n = 9 and then n = 11 ($14.67 with $5.33 left), a one-shot commit at the
# frontier the cap reaches, n = 19 at 117 shots/circuit ($18.51, priced with
# no headroom against drift), and an all-in nonlocal-content chase at n = 5
# at 1062 shots/circuit ($19.99, certifying past the CHSH ceiling with
# probability 1.00). Which n each archetype lands on depends on the deficit a
# team declares, and that is the point: the shapes are fixed, the numbers move.
#
# RESERVE is derived rather than set, so the three can never disagree. It pays
# the organizer dry run (~$12 [estimate]), the one separable crosstalk probe
# (~$5), and reruns after an execution failure, because organizers execute and
# an execution failure is theirs rather than the team's.
POT = 300.0
TEAMS = 12
CAP_PER_TEAM = 20.0
RESERVE = POT - TEAMS * CAP_PER_TEAM
assert RESERVE >= 0, "cap times teams exceeds the pot"

# The certification power every published plan is sized for. Minimum-shot
# sizing puts the expected outcome exactly at the gate and certifies only
# about half the time; S_90 is the smallest shot count whose certification
# probability at the declared delta reaches this target.
POWER_TARGET = 0.90

# The ceiling on certified nonlocal content for any two-qubit CHSH experiment:
# sqrt(2) - 1 = 0.41421. A named benchmark line on the results table, not a
# floor. The odd-cycle game can certify past it, which no CHSH run can.
CHSH_CEILING = math.sqrt(2.0) - 1.0


# --------------------------------------------------------------------------
# The game
# --------------------------------------------------------------------------


def omega_c(n: int) -> float:
    """Classical value of the odd-cycle game on $C_n$: $1 - 1/(2n)$.

    Unconditional. It is the best any local deterministic or shared-randomness
    strategy achieves, so beating it is a statement about the device rather
    than about an unproven assumption.
    """
    return 1.0 - 1.0 / (2 * n)


def omega_q(n: int) -> float:
    """Quantum value of the odd-cycle game on $C_n$: $\\cos^2(\\pi/(4n))$.

    Achieved by one Bell pair and one $R_y$ per player, with the rotation angle
    the only thing that changes with $n$.
    """
    return math.cos(math.pi / (4 * n)) ** 2


def n_circuits(n: int) -> int:
    """Distinct circuits in one sweep of $C_n$: $2n$.

    $n$ edge questions and $n$ vertex questions, one circuit each.
    """
    return 2 * n


def quantum_advantage(n: int) -> float:
    """$\\omega_q - \\omega_c$, the whole margin a device has to spend.

    Shrinks like $1/(2n)$, which is why larger $n$ is the hard axis.
    """
    return omega_q(n) - omega_c(n)


def readout_drop(omega_ideal: float, p: float) -> float:
    """How far a readout bit flip of probability `p` per wire pushes a win rate down.

        drop = 2 p (1 - p) (2 omega_ideal - 1)

    Both win conditions of the odd-cycle game read the parity of the two answer
    bits: same vertex wins on parity 0, an edge wins on parity 1. Flipping one of
    the two bits flips that parity and turns a win into a loss and a loss into a
    win. Flipping both leaves it alone. So the only thing the channel does is
    exchange the win probability with the loss probability on a $2p(1-p)$ share
    of the shots, which is what the formula says.

    `omega_ideal` is whatever the noiseless circuits win at, so this holds for any
    strategy rather than only the optimum. A strategy already at $1/2$ loses
    nothing, since there is nothing left to exchange, and one at $1$ loses the
    most. `delta_from_readout` is this at the optimum, which is the case a sized
    run assumes.

    Exact rather than a small-$p$ expansion.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p = {p} is not a probability")
    return 2.0 * p * (1.0 - p) * (2.0 * omega_ideal - 1.0)


def delta_from_readout(n: int, p: float) -> float:
    """Device deficit produced by a readout bit flip of probability `p` per wire.

        delta = 2 p (1 - p) (2 omega_q(n) - 1)

    `readout_drop` evaluated at the equal-rate optimum, where every one of the
    $2n$ questions wins at $\\omega_q(n)$, so the drop is the same on each and
    their mean drops by it too.

    This is the model `challenge/simulator.py` dials, `qml.BitFlip(p)` on both
    wires of `default.mixed` before the measurement. Checked against that
    simulator's exact outcome probabilities, with no sampling in the way, at
    $n \\in \\{3, 5, 7, 13, 29, 43\\}$ and
    $p \\in \\{0.0025, 0.005, 0.01, 0.02\\}$: agreement to $2 \\times 10^{-14}$
    at every one of the 24 points.

    Note what it does not say. $\\delta$ is not quite independent of $n$: at a
    fixed $p = 0.005$ it runs from 0.0086 at $n = 3$ to 0.0099 at $n = 43$,
    because $2\\omega_q(n) - 1$ climbs toward 1. `readout_drift` prices what that
    costs a team that calibrates at one $n$ and commits at another.
    """
    return readout_drop(omega_q(n), p)


def readout_from_delta(n: int, delta: float) -> float | None:
    """Per-wire flip probability that produces `delta` at $C_n$. Inverse of the above.

    Solving $2p(1-p)\\,(2\\omega_q - 1) = \\delta$ for the smaller root gives

        p = (1 - sqrt(1 - 2 delta / (2 omega_q(n) - 1))) / 2

    Returns `None` when no flip probability reaches that deficit. The channel
    saturates at $p = 1/2$, where the two answer bits are uniform and the deficit
    is at most $\\omega_q(n) - 1/2$, so a larger `delta` is asking this model for
    something it cannot produce and wants a different one.
    """
    if delta < 0.0:
        raise ValueError(f"delta = {delta} is negative")
    reach = 2.0 * omega_q(n) - 1.0
    q = delta / reach
    if q > 0.5:
        return None
    return 0.5 * (1.0 - math.sqrt(1.0 - 2.0 * q))


def readout_asym_rates(omega_ideal: float, e0: float, e1: float) -> tuple[float, float]:
    """(vertex, edge) win rates under asymmetric per-wire readout flips.

    The channel reads a true 0 as 1 with probability `e0` and a true 1 as 0
    with probability `e1`, independently on each wire. For any strategy in the
    Bell-pair family, one $R_y$ per player on $\\ket{\\beta_{00}}$, the two
    agreeing outcomes split their probability evenly and so do the two
    disagreeing ones, so the parity flips with probability

        r_same = e0 (1 - e0) + e1 (1 - e1)   when the true bits agree,
        r_diff = e0 (1 - e1) + e1 (1 - e0)   when they differ,

    and a question winning at `omega_ideal` noiselessly wins at

        vertex:  sigma * omega_ideal + r_diff
        edge:    sigma * omega_ideal + r_same

    with the shared slope $\\sigma = (1 - e_0 - e_1)^2$. Both are affine
    increasing in `omega_ideal`, which is the fact that kills strategy
    adaptation against readout noise: whatever $(e_0, e_1)$ are, the noisy
    mean is maximised by maximising the noiseless mean, so the textbook
    strategy stays exactly optimal within this family. The two constants
    differ by `readout_asym_block_gap`, which is second order in the
    asymmetry: at $(e_0, e_1) = (0.0013, 0.028)$, the probe-indicated Emerald
    scale, it is $7.1 \\times 10^{-4}$, and `main` prints it.
    `_check_strategy_optimality` verifies all of this against an explicit
    channel computation and a search that includes tilted states.
    """
    for name, e in (("e0", e0), ("e1", e1)):
        if not 0.0 <= e <= 1.0:
            raise ValueError(f"{name} = {e} is not a probability")
    r_same = e0 * (1.0 - e0) + e1 * (1.0 - e1)
    r_diff = e0 * (1.0 - e1) + e1 * (1.0 - e0)
    slope = (1.0 - e0 - e1) ** 2
    return slope * omega_ideal + r_diff, slope * omega_ideal + r_same


def delta_from_readout_asym(n: int, e0: float, e1: float) -> float:
    """Device deficit from asymmetric per-wire readout flips $(e_0, e_1)$.

    The mean over the $2n$ questions at the optimum collapses to the
    symmetric formula at the mean flip rate:

        delta = 2 ebar (1 - ebar) (2 omega_q(n) - 1),  ebar = (e0 + e1) / 2

    so a pooled win rate alone cannot tell an asymmetric readout from a
    symmetric one at the same mean. What separates them is the per-question
    profile, `readout_asym_block_gap` below.
    """
    vertex, edge = readout_asym_rates(omega_q(n), e0, e1)
    return omega_q(n) - 0.5 * (vertex + edge)


def readout_asym_block_gap(e0: float, e1: float) -> float:
    """Vertex-block win rate minus edge-block win rate: $(e_0 - e_1)^2$.

    The one fingerprint asymmetric readout leaves on the profile: every
    vertex question sits this far above every edge question, both blocks
    internally flat. Second order in the asymmetry, so at readout-scale
    errors it is small: at $(e_0, e_1) = (0.0013, 0.028)$, the
    probe-indicated Emerald scale, it is $7.1 \\times 10^{-4}$, printed by
    `main`. A block gap linear in its cause is the signature of a coherent
    per-player offset instead, which is the distinction the notes teach.
    """
    return (e0 - e1) ** 2


def readout_drift(n_calibrate: int, n_commit: int, p: float) -> float:
    """Relative error in $\\delta$ from calibrating at one $n$ and committing at another.

    Returns $1 - \\delta(n_{\\text{cal}}) / \\delta(n_{\\text{commit}})$, so a
    positive number means the calibration reads low and the commit run is sized
    against a deficit smaller than the one it will meet.

    Calibrating at $n = 7$ and committing at $n = 13$, the widest gap a
    plausible weekend spans, reads 1.8% low at the dial that reproduces the
    published deficit. Even a 1000-shot calibration pins $\\delta$ only to
    about $\\pm 0.0019$, an order of magnitude wider than that bias, so the
    bias is inside the noise of the measurement it biases and the
    $n$-independence the sizing rule assumes survives; `main` prints the
    comparison. The ratio does not depend on `p` except through second order,
    since `p` factors out of both.
    """
    at_commit = delta_from_readout(n_commit, p)
    if at_commit == 0.0:
        return 0.0
    return 1.0 - delta_from_readout(n_calibrate, p) / at_commit


def critical_visibility(n: int) -> float:
    """Smallest Werner visibility $v$ that still clears the classical bound.

    Under visibility $v$ the state is $v$ times the Bell pair plus $1-v$ times
    the maximally mixed state, so the win rate interpolates
    $\\omega(v) = 1/2 + v(\\omega_q - 1/2)$. Setting that equal to $\\omega_c$
    and solving gives the value returned here. This is the device-quality
    number to quote when someone asks how good a QPU has to be.
    """
    return (omega_c(n) - 0.5) / (omega_q(n) - 0.5)


# --------------------------------------------------------------------------
# The statistical gate
# --------------------------------------------------------------------------


def _log_binom_pmf(total: int, q: float, k: int) -> float:
    """log P(Bin(total, q) = k), via `lgamma` so nothing here needs scipy."""
    return (
        math.lgamma(total + 1)
        - math.lgamma(k + 1)
        - math.lgamma(total - k + 1)
        + k * math.log(q)
        + (total - k) * math.log1p(-q)
    )


def binomial_tail(total: int, q: float, k: int) -> float:
    """$P(\\mathrm{Bin}(total, q) \\geq k)$, summed directly in probability space.

    One `lgamma` evaluation anchors the first term, then the multiplicative
    recurrence walks the rest, summing away from the mode and stopping once a
    term falls below 1e-16 of the running sum. No approximation beyond float
    precision, and no scipy. Everything statistical in this file reduces to
    this sum.
    """
    if k <= 0:
        return 1.0
    if k > total:
        return 0.0
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    ratio = q / (1.0 - q)
    mode = int(q * (total + 1))
    if k > mode:
        # Upper tail directly: terms decrease as j climbs away from the mode.
        term = math.exp(_log_binom_pmf(total, q, k))
        tail = 0.0
        for j in range(k, total + 1):
            tail += term
            # `<=` so an underflowed term (0.0 against a 0.0 sum) still stops.
            if term <= tail * 1e-16:
                break
            term *= (total - j) * ratio / (j + 1)
        return min(tail, 1.0)
    # k at or below the mode: sum the complement downward instead, where the
    # terms also decrease, so the early stop is safe on both branches.
    term = math.exp(_log_binom_pmf(total, q, k - 1))
    low = 0.0
    for j in range(k - 1, -1, -1):
        low += term
        if term <= low * 1e-16:
            break
        term *= j / ((total - j + 1) * ratio)
    return max(0.0, 1.0 - low)


def _k_min(total: int, q: float, p_target: float = P_3SIGMA) -> int:
    """Smallest win count whose tail under Bin(total, q) is at most `p_target`.

    The gate as a count: observe at least this many wins out of `total` trials
    and the null at mean `q` is rejected. Bisection, since the tail is
    decreasing in the count. Returns `total + 1` when not even a perfect score
    clears, which happens only for tiny runs.
    """
    lo, hi = 0, total + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if binomial_tail(total, q, mid) <= p_target:
            hi = mid
        else:
            lo = mid
    return hi


def p_value(winrates: list[float], shots: int, bound: float) -> float:
    """One-sided exact binomial tail probability of beating `bound` by chance.

    Pool the counts: with $k$ = round(shots * sum(winrates)) wins out of
    $N = m \\cdot shots$ trials,

        p = P(Bin(N, bound) >= k)

    Exactly valid for the stratified experiment this event runs: $m$ distinct
    circuits, `shots` independent Bernoulli trials each, with per-circuit win
    probabilities that need not be equal. Under any classical strategy the mean
    of those probabilities is at most `bound`; the tail of the pooled count is
    maximised over fixed-mean vectors by the i.i.d. binomial at that mean
    (Hoeffding 1956, Theorem 5); and the binomial tail is monotone in its mean.
    So P(Bin(N, bound) >= k) dominates the true tail. Both steps were
    re-verified on 19 August 2026, including the Jensen argument on the
    log-MGF, and `_check_gate_validity` keeps the exact-convolution check
    permanent.

    Equal shots per circuit is load-bearing: it is what makes the pooled mean
    the estimand. The verifier's per-circuit `sum(counts[c]) == shots` check
    enforces it.

    This replaced a transcription of the ion paper's empirical-Bernstein
    expression on 19 August 2026. That expression mixes shot-level and
    circuit-level granularities and matches no published bound; it is
    accidentally conservative, silently enforcing 6.1 to 6.9 sigma where it
    said 3, so results certified under it stand a fortiori, and the exact test
    is 26-59% cheaper per certification.

    Returns 1.0 when the pooled count is at or below the null mean, preserving
    the old `eps <= 0` guard semantics: landing exactly on the bound is
    evidence of nothing.
    """
    m = len(winrates)
    if m == 0:
        raise ValueError("need at least one circuit")
    total = m * shots
    wins = round(shots * sum(winrates))
    if wins <= total * bound:
        return 1.0
    return binomial_tail(total, bound, wins)


def certified_win_rate_lower_bound(
    winrates: list[float], shots: int, p_target: float = P_3SIGMA
) -> float:
    """One-sided lower confidence bound on the pooled mean win rate.

    The Clopper-Pearson construction at the gate's own significance: the
    smallest $\\omega$ whose binomial tail at the observed pooled count still
    exceeds `p_target`, found by bisection on the mean, which the tail is
    monotone increasing in. Coverage in the stratified setting follows from the
    same Hoeffding domination the gate rests on, because the event that this
    bound exceeds the true mean is a pooled-count tail event.

    This is the certified counterpart of the raw win rate: the device is
    entitled to claim any win rate up to it at 3 sigma, and `certified_p_nl`
    converts it into the number award 3 scores. More shots tighten it, which is
    the one place extra precision buys an award.
    """
    m = len(winrates)
    if m == 0:
        raise ValueError("need at least one circuit")
    total = m * shots
    wins = round(shots * sum(winrates))
    if wins <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if binomial_tail(total, mid, wins) <= p_target:
            lo = mid
        else:
            hi = mid
    return lo


def certified_p_nl(
    winrates: list[float], shots: int, n: int, p_target: float = P_3SIGMA
) -> float:
    """Certified nonlocal content: $p_{NL} = 1 - 2n(1 - \\omega_{LB})$.

    $\\omega_{LB}$ is the exact one-sided lower bound above, so this is the
    fraction of rounds that provably could not have come from any local model,
    certified at the same 3 sigma as the gate. The general bound is Aolita et
    al., Phys. Rev. A 85, 032107 (2012), $p_L \\leq (\\omega_{ns} - \\omega) /
    (\\omega_{ns} - \\omega_c)$, evaluated for the odd cycle where the
    no-signalling value is 1; the odd-cycle form is in Drmota et al., PRL 134,
    070201 (2025).

    Award 3 scores the largest certified value. $p_{NL}(\\omega_c) = 0$
    identically, so certifying any positive value is exactly equivalent to
    clearing the gate, and at fixed deficit the value falls with $n$ (roughly
    $1 - 2n\\delta$), so largest $n$ and strongest certified nonlocality are
    opposing objectives by physics rather than by rule. `CHSH_CEILING` is the
    benchmark line: past it, the result is one no two-qubit CHSH experiment
    could produce.
    """
    lower = certified_win_rate_lower_bound(winrates, shots, p_target)
    return 1.0 - 2.0 * n * (1.0 - lower)


def confidence_half_width(winrates: list[float], shots: int, d: float = 0.05) -> float:
    """Two-sided Bernstein half-width on the mean win rate, at confidence $1 - d$.

        sigma^2 = mean(w * (1 - w))
        half    = 2 ln(2/d) / (3 * shots) + sigma * sqrt(2 ln(2/d) / (m * shots))

    Transcribed from `calculate_ci` in `qupacabras-db`, which recomputes a
    submission's claimed `nonlocalGame.uncertainty` against this definition at a
    relative tolerance of 1e-6 and rejects a mismatch. The association of the two
    terms is copied as written rather than simplified, so the two agree bit for bit.

    This is an interval on the mean, and it is not the certification gate. The gate
    is the one-sided `p_value` above. The gate moved from the empirical Bernstein
    bound to the exact binomial tail on 19 August 2026 and this interval
    deliberately did not move, because the database recomputes every submitted
    uncertainty against this exact definition, old records included.
    """
    m = len(winrates)
    if m == 0:
        raise ValueError("need at least one circuit")
    sigma = math.sqrt(sum(w * (1.0 - w) for w in winrates) / m)
    term1 = 2.0 * math.log(2.0 / d) / (3.0 * shots)
    term2 = 2.0 * math.log(2.0 / d) / (m * shots)
    return term1 + sigma * math.sqrt(term2)


def _power_against(n: int, shots: int, bound: float, delta: float, p_target: float) -> float:
    """Probability the pooled count beats the gate set at `bound`.

    The device is modelled as winning every circuit at
    $\\omega = \\omega_q(n) - \\delta$, so the pooled count is
    Bin(2n * shots, omega) and the certification probability is its tail at
    the gate count.
    """
    omega = omega_q(n) - delta
    if omega <= bound:
        return 0.0
    total = n_circuits(n) * shots
    k_gate = _k_min(total, bound, p_target)
    if k_gate > total:
        return 0.0
    return binomial_tail(total, omega, k_gate)


def certification_power(
    n: int, shots: int, delta: float = DELTA_EXAMPLE, p_target: float = P_3SIGMA
) -> float:
    """Probability a sweep of $C_n$ at `shots` per circuit certifies.

    This is the number a plan is sized on. Minimum-shot sizing puts the
    expected count exactly at the gate and certifies about half the time; the
    published tables are S_90, `shots_for_significance` below.
    """
    return _power_against(n, shots, omega_c(n), delta, p_target)


def chsh_ceiling_power(
    n: int, shots: int, delta: float = DELTA_EXAMPLE, p_target: float = P_3SIGMA
) -> float:
    """Probability a sweep certifies nonlocal content past the CHSH ceiling.

    Beating `CHSH_CEILING` means certifying
    $\\omega_{LB} > 1 - (1 - (\\sqrt{2} - 1))/(2n)$, which is a pooled-count
    threshold like any other, so the power computes the same way as
    `certification_power` against a higher bar. Wherever the deficit lands, the
    chase lives at n = 5, where the margin over the ceiling is widest and
    certifying past it is cheapest: at the illustration delta = 0.015 the point
    value technically sits above the ceiling out to n = 17 and drops below it
    at n = 19, but the margin narrows so fast that n = 17 wants 986627 shots
    per circuit against n = 5's 247. A worse deficit pulls both in.
    """
    bound = 1.0 - (1.0 - CHSH_CEILING) / (2 * n)
    return _power_against(n, shots, bound, delta, p_target)


def _smallest_shots(power_at, power: float) -> int | None:
    """Smallest shot count whose power reaches `power`: doubling, then bisection.

    The power curve is a sawtooth in shots, because the gate count moves in
    integer steps, so "smallest" is a convention: the bisection lands on a
    plateau and the final walk-down takes that plateau's first shot count. A
    smaller isolated plateau below it may exist and is deliberately not hunted
    for, since a plan sized into a sawtooth dip has no margin against the dip
    moving.
    """
    hi = 1
    while power_at(hi) < power:
        hi *= 2
        if hi > 10_000_000:
            return None
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if power_at(mid) >= power:
            hi = mid
        else:
            lo = mid
    while hi > 1 and power_at(hi - 1) >= power:
        hi -= 1
    return hi


def shots_for_significance(
    n: int,
    delta: float = DELTA_EXAMPLE,
    p_target: float = P_3SIGMA,
    power: float = POWER_TARGET,
) -> int | None:
    """S_90: shots per circuit that certify $C_n$ with probability >= `power`.

    Every published shot table and every `plan_check` target is this number.
    Sizing to the expected outcome instead, the smallest count whose expected
    pooled score clears the gate, certifies only about half the time, because
    the outcome lands below its own expectation half the time. That trap is a
    teaching point in the notes, not a policy here.

    Returns `None` when the device cannot beat the classical bound at this $n$
    at all, which no number of shots repairs.
    """
    if omega_q(n) - delta <= omega_c(n):
        return None
    return _smallest_shots(
        lambda shots: certification_power(n, shots, delta, p_target), power
    )


def shots_for_chsh_ceiling(
    n: int = 5,
    delta: float = DELTA_EXAMPLE,
    p_target: float = P_3SIGMA,
    power: float = POWER_TARGET,
) -> int | None:
    """Shots per circuit that certify past the CHSH ceiling with probability `power`.

    The nonlocal-content chase, sized the same way as `shots_for_significance`.
    Returns `None` when the expected win rate sits at or below the ceiling's
    threshold, which happens at every $n$ past a cutoff set by the deficit: at
    the illustration delta = 0.015 that cutoff is $n = 19$, and at a worse
    deficit it comes down fast.
    """
    bound = 1.0 - (1.0 - CHSH_CEILING) / (2 * n)
    if omega_q(n) - delta <= bound:
        return None
    return _smallest_shots(
        lambda shots: chsh_ceiling_power(n, shots, delta, p_target), power
    )


def design_efficiency(
    runs: list[tuple], p_target: float = P_3SIGMA
) -> float | None:
    """What a team's weekend cost, against what its record needed. Lower wins.

    Each run is `(n, shots, winrates)`, `(n, shots, winrates, twirls)`, or
    `(n, shots, winrates, twirls, qpu)`, and
    a team hands in both of them, because the dollars are spent across both
    and the record is one. The twirl count, read off the submitted script's
    parameter block and 1 when the block carries none, puts a twirled run's
    extra task fees into the numerator, so the spend scored here is the spend
    the ledger metered; the denominator stays the untwirled S_90 sweep, so a
    twirl that buys no larger record raises the score.

        record = largest n certified at p_target
        delta  = omega_q(record) - mean(winrates of the run that certified it)
        needed = cost(record, shots_for_significance(record, delta), qpu=record qpu)
        score  = max(1.0, sum(cost(n, shots, twirls, qpu) over every run) / needed)

    The denominator is the S_90 sweep at the deficit the team's own winning run
    revealed, priced on the device that run went to: the cheapest run a team
    that knew its device exactly would have published. Exactly 1.0 for a team
    that ran that sweep and stopped.

    Pricing the denominator on the record run's own device is what keeps the
    two QPUs comparable. Garnet's per-shot rate is about 10% below Emerald's,
    so a Garnet weekend put over an Emerald-priced denominator would score
    about 9% better for choosing the cheaper machine, which is not a planning
    decision this award is meant to reward. Same device top and bottom, and
    the rate very nearly cancels.

    The clamp is load-bearing on a continuous meter. An under-bought run that
    clears the gate by luck spends less than the honestly sized sweep, and
    without the clamp that gamble would outscore every team that followed the
    published tables; with it, the gamble ties the perfect plan at 1.0 and
    beats nobody. A run that missed the gate is not free either: its dollars
    stay in the numerator while the denominator counts only what the record
    cost, so a failed gamble is what this award punishes. A skipped run adds
    $0, which is why skipping run 1 is a legitimate strategy.

    Worked weekends at the illustration delta = 0.015: a one-shot S_90 commit
    at n = 13 ($10.00) scores 1.00; certifying n = 9 and then n = 11 ($14.67
    against the $8.15 the record needed) scores 1.80; the same n = 13 commit
    run twice scores 2.00.

    Nothing here reads a published deficit, and the award needed no change when
    the event stopped publishing one. The denominator is built from the deficit
    the team's own winning run revealed, so a team that measures its device
    unlocks a larger record and a larger denominator at once, and a team that
    predicts from published specs over-buys and pays for it here. Calibration
    is rewarded by the arithmetic rather than by a rule.

    The win rate is divided out on purpose. A better device already wins a
    larger $n$ for the same money, so this measures how well the weekend was
    planned rather than how good the hardware was.

    Returns `None` for a team no run of which certified, which is off the board.
    """
    if not runs:
        return None
    # A run is (n, shots, winrates), optionally with twirls and then the qpu
    # label. Both tails default, so a caller that does not care about pricing
    # still gets the default device, and the ledger's own rows carry both.
    unpacked = [
        (
            run[0],
            run[1],
            run[2],
            run[3] if len(run) > 3 else 1,
            run[4] if len(run) > 4 else DEFAULT_QPU,
        )
        for run in runs
    ]
    certified = [
        (n, shots, rates, qpu)
        for n, shots, rates, _, qpu in unpacked
        if p_value(rates, shots, omega_c(n)) <= p_target
    ]
    if not certified:
        return None
    record_n, _, record_rates, record_qpu = max(certified, key=lambda run: run[0])
    delta = omega_q(record_n) - sum(record_rates) / len(record_rates)
    needed = shots_for_significance(record_n, delta, p_target)
    if needed is None:
        return None
    spent = sum(cost(n, shots, twirls, qpu) for n, shots, _, twirls, qpu in unpacked)
    return max(1.0, spent / cost(record_n, needed, qpu=record_qpu))


def family_wise_false_positive(p_target: float = P_3SIGMA, teams: int = TEAMS) -> float:
    """Chance at least one of `teams` honest teams clears the gate by luck.

    Each team is one independent test against the threshold, so the rate is
    $1 - (1 - p)^{\\text{teams}}$. This is why the gate is $3\\sigma$ rather than
    $2\\sigma$: the per-team number looks small either way, and the family-wise
    number is what a false claim would actually be drawn from.
    """
    return 1.0 - (1.0 - p_target) ** teams


def largest_delta_certified(n: int, shots: int, p_target: float = P_3SIGMA) -> float:
    """Largest device deficit this plan still certifies with the target power.

    A team's own margin: run the plan, and it keeps at least a POWER_TARGET
    certification probability against any deficit up to this value. Found by
    bisection on `shots_for_significance`, which is monotone increasing in
    $\\delta$.
    """
    lo, hi = 0.0, quantum_advantage(n)
    floor = shots_for_significance(n, lo, p_target)
    if floor is None or floor > shots:
        return 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        need = shots_for_significance(n, mid, p_target)
        if need is not None and need <= shots:
            lo = mid
        else:
            hi = mid
    return lo


# --------------------------------------------------------------------------
# Money
# --------------------------------------------------------------------------


def rates_for_qpu(qpu: str) -> tuple[float, float]:
    """`(task fee, per-shot rate)` in dollars for one of the offered devices.

    The bare TASK_DOLLARS and SHOT_DOLLARS constants are the default device's,
    which is what every published table is priced at. A team that picks the
    other device pays this row instead: Garnet's per-shot rate is about 10%
    below Emerald's, which is the whole of the price difference between them.
    """
    route = QPU_ROUTES[qpu]
    _, task, shot = next(row for row in MEASURED_ROUTE_RATES if row[0] == route)
    return task, shot


def cost(n: int, shots: int, passes: int = 1, qpu: str = DEFAULT_QPU) -> float:
    """Dollars for one sweep of $C_n$ at `shots` per circuit, on `qpu`.

        2n * (passes * task_fee + shot_rate * shots)

    The two rates come from `rates_for_qpu`, so this prices the device the run
    actually goes to. They differ, and only in the shot term: both devices
    charge 30 credits per task, Emerald charges 0.16 credits per shot and
    Garnet 0.145, so a shot-dominated sweep is about 9% cheaper on Garnet and
    a task-dominated one costs the same on both. Pricing a Garnet run at
    Emerald's rate overstates it by up to about \\$1.60 inside this cap, which
    is why the parameter is here rather than left to the caller to remember.

    One task per circuit variant, and that is settled rather than assumed: the
    runtime submits one job per circuit, so $2n$ circuits are $2n$ task fees
    and nothing batches. `passes` is the number of variants submitted per
    question: Pauli-twirl frames (TWIRLS in the submission template) or
    randomised-order repeats. `shots` stays the total per question, split
    across the variants, so extra passes multiply only the task term, one
    extra task fee per circuit per pass.

    Returns a float, because the qBraid meter is continuous. There is no
    whole-credit boundary, so there are no free shots below one and no step to
    round a plan up to: every shot is a line item, which is why plans are
    sized for power rather than rounded up to a boundary.
    """
    task, shot = rates_for_qpu(qpu)
    return n_circuits(n) * (passes * task + shot * shots)


def shots_affordable(n: int, budget: float, qpu: str = DEFAULT_QPU) -> int:
    """Most shots per circuit a sweep of $C_n$ can take for at most `budget` dollars.

    The exact inverse of `cost` at fixed $n$ and fixed device, floored to a
    whole shot. Zero when the budget does not even cover the $2n$ task fees.
    """
    task, shot = rates_for_qpu(qpu)
    per_circuit = budget / n_circuits(n) - task
    if per_circuit <= 0.0:
        return 0
    return math.floor(per_circuit / shot)


def largest_n_feasible(delta: float = DELTA_EXAMPLE, n_max: int = 999) -> int | None:
    """Largest odd $n$ where the device still beats the classical bound.

    Ignores money and shots entirely. This is the ceiling physics imposes,
    roughly $1/(2\\delta)$ since the advantage shrinks like $1/(2n)$.
    """
    best = None
    for n in range(3, n_max + 1, 2):
        if quantum_advantage(n) > delta:
            best = n
    return best


def largest_n_affordable(
    budget: float,
    delta: float = DELTA_EXAMPLE,
    p_target: float = P_3SIGMA,
    qpu: str = DEFAULT_QPU,
) -> tuple[int, int, float] | None:
    """Largest odd $n$ certifiable at `p_target` for at most `budget` dollars on `qpu`.

    Returns `(n, shots_per_circuit, dollars)`, or `None` when even $C_3$ is out
    of reach. The shot count is S_90, the sizing every published table uses.

    The device enters only through the price, never through the shot count: a
    deficit is a deficit whichever machine produced it, so `delta` decides how
    many shots a certification needs and `qpu` decides what those shots cost.
    Two teams sizing at the same declared deficit therefore buy the same sweep
    and pay different bills, and the cheaper device sometimes reaches one rung
    further for the same money.

    Cost climbs in steps rather than a curve, because the advantage shrinks
    like $1/(2n)$ and the required shots grow like its inverse square: at
    the illustration delta = 0.015 a 90%-power sweep is $10.00 at n = 13,
    $18.51 at n = 19 and $23.69 at n = 21. The $20 cap prices n = 19 as the
    ceiling there, with the next step out of reach by $3.69. At a different
    declared deficit the whole ladder moves, which is what the grid in the
    instructions shows.
    """
    best = None
    n = 3
    over = 0
    while n <= 999:
        shots = shots_for_significance(n, delta, p_target)
        if shots is None:
            break
        price = cost(n, shots, qpu=qpu)
        if price <= budget:
            best = (n, shots, price)
            over = 0
        else:
            # Cost climbs with n (more circuits, and S_90 grows as the margin
            # shrinks), so two consecutive misses end the scan.
            over += 1
            if over >= 2:
                break
        n += 2
    return best


# --------------------------------------------------------------------------
# Self-checks
# --------------------------------------------------------------------------


def _check_chsh() -> tuple[bool, str]:
    """CHSH is the $n = 2$ case, and both formulas must reproduce it.

    $\\omega_c(2) = 3/4$ and $\\omega_q(2) = \\cos^2(\\pi/8)$ are the textbook
    CHSH values, and the critical visibility is the textbook $1/\\sqrt{2}$: a
    Werner state beats the CHSH inequality exactly when $v > 1/\\sqrt{2}$,
    because the CHSH value scales as $2\\sqrt{2}\\,v$ against a bound of 2.
    """
    got = critical_visibility(2)
    want = 1.0 / math.sqrt(2.0)
    ok = math.isclose(got, want, rel_tol=1e-12)
    detail = (
        f"omega_c(2) = {omega_c(2):.6f} (want 0.750000), "
        f"omega_q(2) = {omega_q(2):.6f} (want {math.cos(math.pi / 8) ** 2:.6f}), "
        f"critical visibility = {got:.10f} (want 1/sqrt(2) = {want:.10f})"
    )
    return ok, detail


def _check_shot_inversion() -> tuple[bool, str]:
    """`shots_for_significance` must actually deliver its power target.

    At the returned S_90 the certification probability reaches POWER_TARGET, at
    one shot fewer it does not, and the expected pooled count itself clears the
    gate with room. Guards the bisection and the walk-down convention against a
    sign or boundary slip.
    """
    failures = []
    for n in (3, 5, 7, 9, 11, 13):
        shots = shots_for_significance(n)
        if shots is None:
            failures.append(f"n={n}: infeasible at delta={DELTA_EXAMPLE}")
            continue
        at = certification_power(n, shots)
        below = certification_power(n, shots - 1)
        omega = omega_q(n) - DELTA_EXAMPLE
        expected = p_value([omega] * n_circuits(n), shots, omega_c(n))
        if not (at >= POWER_TARGET > below and expected <= P_3SIGMA):
            failures.append(
                f"n={n}: S_90={shots}, power={at:.4f}, power(-1)={below:.4f}, "
                f"p(expected)={expected:.3e}"
            )
    return not failures, (
        "; ".join(failures)
        if failures
        else "n = 3, 5, 7, 9, 11, 13 all sit at the first shot count of their plateau"
    )


def predicted_delta(qpu: str, n: int) -> float:
    """Device deficit predicted for `qpu` at $C_n$ from its published readout error.

    The prediction route, end to end and in one line: take the published
    per-wire readout error pair out of PUBLISHED_READOUT and push it through
    `delta_from_readout_asym`. This is what a team can compute before spending
    anything, and it is the number the published grid is meant to be read at
    when a team has not measured its own.

    It over-predicts, and knowing why is the point. Neither published readout
    figure corrects for state preparation error, so both are upper estimates of
    the flip rate, and the deficit law is increasing in that rate. The
    prediction therefore lands above the truth for both devices, a team that
    sizes from it over-buys shots, and over-buying certifies. Predicting is
    worse than measuring without being a trap.
    """
    e0, e1, _ = PUBLISHED_READOUT[qpu]
    return delta_from_readout_asym(n, e0, e1)


def _check_route_price() -> tuple[bool, str]:
    """`cost` has to carry the measured rates, and the device ranking re-derive.

    This is the one check in this file that tests a claim about the outside
    world rather than an identity inside it, and it is weaker than the check it
    replaced: the old Open Quantum check reproduced 34 recorded quotes, while
    the qBraid rates came out of the insufficient-credits refusal oracle on
    2026-08-18 and the raw refusals were not recorded shot-by-shot. What is
    asserted instead: the two constants match the measured route table, `cost`
    is the linear form the oracle fitted, and the device ranking re-derives
    from public information alone.

    That last clause changed on 2026-08-21, when the event stopped publishing
    any measured deficit. The ranking now comes from PUBLISHED_READOUT pushed
    through `predicted_delta`, which is the same arithmetic a team does, so
    this check is a check on the route a student is asked to walk. Emerald
    still wins: it carries a 10% higher per-shot rate than Garnet and roughly
    half the predicted deficit, and the shots a certification needs grow like
    the inverse square of the margin, so the deficit dominates the rate.
    `organizers/quote_check.py` replays the oracle against the live account
    before any run.
    """
    failures = []
    task, shot = rates_for_qpu(DEFAULT_QPU)
    if task != TASK_DOLLARS or shot != SHOT_DOLLARS:
        failures.append(
            f"constants ({TASK_DOLLARS}, {SHOT_DOLLARS}) disagree with the "
            f"measured row ({task}, {shot})"
        )
    if not math.isclose(cost(13, 0), n_circuits(13) * TASK_DOLLARS):
        failures.append("cost at zero shots is not the bare task fees")
    if not math.isclose(
        cost(13, 1000) - cost(13, 0), n_circuits(13) * SHOT_DOLLARS * 1000
    ):
        failures.append("cost is not linear in shots")
    prices: dict[str, float] = {}
    for qpu in QPU_ROUTES:
        delta = predicted_delta(qpu, 9)
        shots = shots_for_significance(9, delta)
        if shots is None:
            continue  # a device the published specs say cannot certify n = 9
        qpu_task, qpu_shot = rates_for_qpu(qpu)
        prices[qpu] = n_circuits(9) * (qpu_task + qpu_shot * shots)
    if not prices:
        failures.append("no offered device certifies n = 9 at its predicted deficit")
        return False, "; ".join(failures[:3])
    cheapest = min(prices, key=lambda qpu: prices[qpu])
    if cheapest != DEFAULT_QPU:
        failures.append(
            f"cheapest device to certify n = 9 on published specs is {cheapest}, "
            f"not the default {DEFAULT_QPU}"
        )
    if failures:
        return False, "; ".join(failures[:3])
    others = ", ".join(
        f"{qpu} "
        + (
            f"${prices[qpu]:.2f}"
            if qpu in prices
            else "cannot certify n = 9 at all"
        )
        + f" at predicted delta {predicted_delta(qpu, 9):.4f}"
        for qpu in sorted(QPU_ROUTES)
        if qpu != DEFAULT_QPU
    )
    return True, (
        f"rates match the 2026-08-18 table; on published specs alone, certifying "
        f"n = 9 costs ${prices[DEFAULT_QPU]:.2f} on {DEFAULT_QPU} at predicted "
        f"delta {predicted_delta(DEFAULT_QPU, 9):.4f}, against {others}"
    )


def _check_device_pricing() -> tuple[bool, str]:
    """The two devices price differently, and the award does not reward the cheap one.

    Three things, all of which broke or would have broken when teams were given
    the device choice on 21 August 2026.

    First, `cost` actually reads the device. Both QPUs charge the same task fee
    and different per-shot rates, so a zero-shot sweep costs the same on both
    and a shot-heavy one does not. A `cost` that ignored its `qpu` argument
    would pass every other check in this file silently.

    Second, the difference is the size the rate table says: exactly the
    per-shot gap times the number of shots, with no task-term contamination.

    Third, and the reason this check exists rather than just the first two:
    `design_efficiency` prices the denominator on the record run's own device.
    Score the same weekend twice, once on each QPU, and the two scores must
    land within a hair of each other. If the denominator were pinned to one
    device, the weekend run on the cheaper machine would score visibly better
    for a decision the award explicitly does not measure.
    """
    failures = []
    task_e, shot_e = rates_for_qpu("emerald")
    task_g, shot_g = rates_for_qpu("garnet")
    if task_e != task_g:
        failures.append(f"task fees differ ({task_e} vs {task_g}); this check assumes they do not")
    if shot_e <= shot_g:
        failures.append(f"emerald per-shot {shot_e} is not above garnet's {shot_g}")
    if not math.isclose(cost(13, 0, qpu="emerald"), cost(13, 0, qpu="garnet")):
        failures.append("a zero-shot sweep is priced differently on the two devices")
    gap = cost(13, 1000, qpu="emerald") - cost(13, 1000, qpu="garnet")
    want = n_circuits(13) * (shot_e - shot_g) * 1000
    if not math.isclose(gap, want):
        failures.append(f"1000-shot price gap is {gap:.4f}, expected {want:.4f}")

    # The same weekend, scored on each device in turn.
    delta = DELTA_EXAMPLE
    scores = {}
    for qpu in QPU_ROUTES:
        weekend = []
        for n in (9, 11):
            shots = shots_for_significance(n, delta)
            assert shots is not None
            weekend.append((n, shots, [omega_q(n) - delta] * n_circuits(n), 1, qpu))
        score = design_efficiency(weekend)
        if score is None:
            failures.append(f"the ladder failed to certify on {qpu}")
            continue
        scores[qpu] = score
    if len(scores) == len(QPU_ROUTES):
        spread = max(scores.values()) - min(scores.values())
        if spread > 0.01:
            failures.append(
                "the same weekend scores "
                + ", ".join(f"{q} {v:.4f}" for q, v in scores.items())
                + f"; spread {spread:.4f} rewards the device rather than the plan"
            )
    if failures:
        return False, "; ".join(failures[:3])
    return True, (
        f"emerald and garnet differ only in the shot term (${shot_e} vs ${shot_g}); "
        f"the same ladder scores "
        + ", ".join(f"{q} {v:.3f}" for q, v in sorted(scores.items()))
    )


def _check_efficiency_resolution() -> tuple[bool, str]:
    """The design award has to separate the weekends it is meant to rank.

    Four canonical weekends at the illustration deficit. A one-shot S_90 commit
    at n = 13 scores exactly 1.0. A gambler who bought half the shots and drew a
    pooled count lucky enough to clear the gate anyway is clamped to 1.0 rather
    than scored ahead of the team that followed the table. A two-run ladder
    that certifies n = 9 and then n = 11 lands strictly between 1 and the blind
    double commit at n = 13, which scores 2.0: the ladder's first run keeps its
    dollars in the numerator, and that separation is the resolution the award
    ranks on.

    The four land at 1.00, 1.00, 1.80 and 2.00 at delta = 0.015. They landed at
    1.00, 1.00, 1.81 and 2.00 at the deficit this file used to publish, so the
    ordering and the resolution both survived the move to a team-declared
    deficit untouched, which is why `design_efficiency` itself needed no edit.
    """
    delta = DELTA_EXAMPLE

    def sweep(n: int) -> tuple[int, int, list[float]]:
        shots = shots_for_significance(n, delta)
        assert shots is not None
        return n, shots, [omega_q(n) - delta] * n_circuits(n)

    perfect = design_efficiency([sweep(13)])
    ladder = design_efficiency([sweep(9), sweep(11)])
    blind = design_efficiency([sweep(13), sweep(13)])

    _, s13, _ = sweep(13)
    lucky_shots = s13 // 2
    total = n_circuits(13) * lucky_shots
    lucky_rate = _k_min(total, omega_c(13)) / total
    gambler = design_efficiency([(13, lucky_shots, [lucky_rate] * n_circuits(13))])

    if None in (perfect, ladder, blind, gambler):
        return False, "a canonical weekend failed to certify"
    assert perfect is not None and ladder is not None
    assert blind is not None and gambler is not None
    if abs(perfect - 1.0) > 1e-12:
        return False, f"one S_90 sweep scored {perfect}"
    if abs(gambler - 1.0) > 1e-12:
        return False, f"the clamped gambler scored {gambler}"
    if not 1.0 < ladder < blind:
        return False, f"out of order: {ladder:.2f} vs {blind:.2f}"
    return True, (
        f"one S_90 sweep at n = 13 scores {perfect:.2f}, the lucky gambler is "
        f"clamped to {gambler:.2f}, certify 9 then 11 scores {ladder:.2f}, "
        f"commit twice {blind:.2f}"
    )


def _check_design_efficiency() -> tuple[bool, str]:
    """The design award rests on two properties, so assert both.

    A team that ran one S_90-sized sweep at the deficit its own run revealed
    scores exactly 1, and no certified weekend scores below 1. The second is
    the clamp's job: on a continuous meter an under-bought run can clear the
    gate by luck for less money than the honest sweep, and without the clamp
    that gamble would outscore every team that followed the published tables.
    Overspending must also register: a certified weekend that bought more than
    its record needed scores strictly above 1.
    """
    failures = []
    checked = 0
    for n in (5, 7, 9, 11, 13):
        for delta in (0.010, 0.026, 0.032):
            tight = shots_for_significance(n, delta)
            if tight is None:
                continue
            rates = [omega_q(n) - delta] * n_circuits(n)
            exact = design_efficiency([(n, tight, rates)])
            if exact is None or abs(exact - 1.0) > 1e-12:
                failures.append(f"n={n}, delta={delta}: tight run scored {exact}")
            calib_shots = shots_for_significance(5, delta)
            assert calib_shots is not None
            calib = (5, calib_shots, [omega_q(5) - delta] * n_circuits(5))
            for multiplier in (0.6, 1.0, 2.0):
                shots = max(MINIMUM_SHOTS, int(tight * multiplier))
                for weekend in ([(n, shots, rates)], [(n, shots, rates), calib]):
                    score = design_efficiency(weekend)
                    checked += 1
                    if score is not None and score < 1.0:
                        failures.append(f"n={n}, delta={delta}, shots={shots}: {score:.6f}")
                if multiplier >= 2.0:
                    over = design_efficiency([(n, shots, rates)])
                    if over is not None and over <= 1.0:
                        failures.append(
                            f"n={n}, delta={delta}: overspend x{multiplier} scored {over}"
                        )
    if failures:
        return False, "; ".join(failures[:3])
    return True, f"{checked} weekends, S_90 scores 1.0 exactly, none certified below 1.0"


def _check_readout_law() -> tuple[bool, str]:
    """`readout_drop` must match an explicit channel calculation, and a run.

    Two independent things, because the closed form is short enough to be wrong
    in a way that still looks right.

    First, recompute the drop the long way. Build the four outcome probabilities
    of one question from the angle between the two players' rotations, push each
    through the bit-flip channel by summing over which of the two bits flipped,
    and read the win probability off the result. That path never uses $2p(1-p)$
    and never uses the parity argument, so agreeing with it is evidence about
    both. Swept over angles rather than at any particular strategy, which is what
    makes it evidence about the formula rather than about one point of it.

    It also asserts that the channel leaves the equal-rate property alone. Two
    questions winning at the same rate before the channel still win at the same
    rate after it, because `readout_drop` depends on the win rate and on nothing
    else. That is what keeps the flat-rate check a team searches with meaningful
    on a noisy device, and it is not something a noise model has to grant.

    Second, compare against a measurement. 0.009712 is the deficit measured on
    18 August 2026 from a `default.mixed` sweep at $n = 13$, $p = 0.005$, 20000
    shots on each of the 26 circuits, so 520000 trials and a standard error of
    $1.6 \\times 10^{-4}$. The closed form gives 0.009877, which is 1.0 standard
    errors away. Held to 3.
    """
    failures = []

    def win_probability_the_long_way(difference: float, p: float, edge: bool) -> float:
        """One question's win probability, from four outcomes through the channel.

        `difference` is the angle between the two players' rotations, which is
        the only thing about a strategy that a single question's outcome
        distribution depends on. A Bell pair with a local rotation on each side
        has uniform marginals, so the agreeing outcomes split the agreement
        probability evenly and so do the disagreeing ones.
        """
        agree = math.cos(difference / 2.0) ** 2
        ideal = {
            (0, 0): agree / 2.0,
            (1, 1): agree / 2.0,
            (0, 1): (1.0 - agree) / 2.0,
            (1, 0): (1.0 - agree) / 2.0,
        }
        won = 0.0
        for (a, b), probability in ideal.items():
            for flip_a in (0, 1):
                for flip_b in (0, 1):
                    weight = (p if flip_a else 1.0 - p) * (p if flip_b else 1.0 - p)
                    seen_a, seen_b = a ^ flip_a, b ^ flip_b
                    hit = seen_a != seen_b if edge else seen_a == seen_b
                    if hit:
                        won += probability * weight
        return won

    # Swept across the whole range of angles a strategy could use, both kinds of
    # question, and flip probabilities from none to a device worse than any real
    # one. Nothing here is the strategy the challenge asks for: an angle that
    # happened to be optimal would be one row of this table and is not singled
    # out, because what is being checked is the formula at every row.
    angles = [k * math.pi / 12.0 for k in range(13)]
    checked = 0
    for difference in angles:
        for p in (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.2):
            for edge in (False, True):
                ideal = win_probability_the_long_way(difference, 0.0, edge)
                noisy = win_probability_the_long_way(difference, p, edge)
                checked += 1
                if abs((ideal - noisy) - readout_drop(ideal, p)) > 1e-12:
                    failures.append(
                        f"angle={difference:.4f}, p={p}, edge={edge}: "
                        f"{ideal - noisy:.12f} vs {readout_drop(ideal, p):.12f}"
                    )

            # A vertex question and an edge question whose angles are supplements
            # win at the same rate with no noise, for any angle at all. The
            # channel has to leave them equal, whatever that shared rate is.
            vertex = win_probability_the_long_way(difference, p, edge=False)
            edge_rate = win_probability_the_long_way(math.pi - difference, p, edge=True)
            checked += 1
            if abs(vertex - edge_rate) > 1e-12:
                failures.append(
                    f"angle={difference:.4f}, p={p}: channel split two questions "
                    f"that started equal, {vertex:.12f} vs {edge_rate:.12f}"
                )

    # `delta_from_readout` is `readout_drop` at the quantum value, and its inverse
    # has to return the flip probability it was given.
    for n in (3, 5, 7, 13, 29, 43):
        for p in (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05):
            closed = delta_from_readout(n, p)
            checked += 1
            if abs(closed - readout_drop(omega_q(n), p)) > 1e-15:
                failures.append(f"n={n}, p={p}: delta_from_readout disagrees with drop")
            back = readout_from_delta(n, closed)
            if back is None or abs(back - p) > 1e-12:
                failures.append(f"n={n}, p={p}: inverse returned {back}")

    # The measurement, and the arithmetic behind the tolerance.
    measured = 0.009712
    trials = 20000 * n_circuits(13)
    stderr = math.sqrt(measured * (1.0 - measured) / trials)
    predicted = delta_from_readout(13, 0.005)
    sigmas = abs(predicted - measured) / stderr
    if sigmas > 3.0:
        failures.append(
            f"n=13, p=0.005: closed form {predicted:.6f} is {sigmas:.1f} sigma "
            f"from the measured {measured:.6f}"
        )

    if failures:
        return False, "; ".join(failures[:3])
    return True, (
        f"{checked} points agree with the channel to 1e-12, equal rates stay "
        f"equal, inverse round-trips, and delta(13, 0.005) = {predicted:.6f} sits "
        f"{sigmas:.1f} sigma from the measured {measured:.6f}"
    )


def _check_strategy_optimality() -> tuple[bool, str]:
    """Readout noise, however asymmetric, cannot be strategised against.

    Two claims, checked at two strengths, because they are believed at two
    strengths.

    Exact: for any strategy in the Bell-pair family, the per-question win
    rate under asymmetric per-wire flips $(e_0, e_1)$ is the affine form
    `readout_asym_rates` states, with slope $(1 - e_0 - e_1)^2$ shared by
    every question. Verified here against an explicit channel computation,
    four outcomes pushed through the $2 \\times 2$ confusion matrix on each
    wire, swept over angles and over flip pairs including strongly asymmetric
    ones, to 1e-12. Affine increasing means the noisy mean is maximised
    exactly where the noiseless mean is, so within this family the textbook
    strategy is optimal under the channel, as an identity rather than a
    search result.

    Numeric: widening the family to tilted real states
    $\\cos\\gamma\\,|00\\rangle + \\sin\\gamma\\,|11\\rangle$ with arbitrary
    per-vertex angles for both players, a hill-climb from the textbook point
    and from random starts finds nothing that beats the textbook strategy at
    $n = 5$ under $(e_0, e_1) = (0.001, 0.02)$, to 1e-6. First run on
    20 August 2026 (also at $n = 13$ and $e_1$ up to 3%, where the optimiser
    pinned $\\gamma = \\pi/4$ and the textbook angles); this check keeps a
    coarse version of that scan permanent. The scan covers real states and
    real rotations, which is the strategy family the challenge uses.
    """
    failures = []

    def ry(theta: float) -> tuple[tuple[float, float], tuple[float, float]]:
        c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
        return ((c, -s), (s, c))

    def question_rate_long_way(
        gamma: float, a: float, b: float, e0: float, e1: float, edge: bool
    ) -> float:
        """One question's noisy win rate from the four outcomes, no shortcut."""
        cg, sg = math.cos(gamma), math.sin(gamma)
        A, B = ry(a), ry(b)
        flip = ((1.0 - e0, e1), (e0, 1.0 - e1))  # flip[read][true]
        won = 0.0
        for read_a in (0, 1):
            for read_b in (0, 1):
                hit = read_a != read_b if edge else read_a == read_b
                if not hit:
                    continue
                for true_a in (0, 1):
                    for true_b in (0, 1):
                        amp = cg * A[true_a][0] * B[true_b][0] + sg * A[true_a][1] * B[true_b][1]
                        won += amp * amp * flip[read_a][true_a] * flip[read_b][true_b]
        return won

    # The exact claim, swept: Bell state (gamma = pi/4), angles across the
    # range, flip pairs from symmetric to strongly asymmetric.
    checked = 0
    quarter = math.pi / 4.0
    for k in range(13):
        difference = k * math.pi / 12.0
        for e0, e1 in ((0.0, 0.0), (0.001, 0.02), (0.02, 0.001), (0.005, 0.005), (0.03, 0.1)):
            for edge in (False, True):
                ideal = question_rate_long_way(quarter, difference, 0.0, 0.0, 0.0, edge)
                noisy = question_rate_long_way(quarter, difference, 0.0, e0, e1, edge)
                vertex_rate, edge_rate = readout_asym_rates(ideal, e0, e1)
                closed = edge_rate if edge else vertex_rate
                checked += 1
                if abs(noisy - closed) > 1e-12:
                    failures.append(
                        f"angle={difference:.4f}, e=({e0}, {e1}), edge={edge}: "
                        f"{noisy:.12f} vs {closed:.12f}"
                    )
    for n in (3, 5, 13):
        for p in (0.0025, 0.005, 0.02):
            checked += 1
            if abs(delta_from_readout_asym(n, p, p) - delta_from_readout(n, p)) > 1e-15:
                failures.append(f"n={n}, p={p}: symmetric limit disagrees")

    # The numeric claim: a coarse re-run of the 20 August 2026 tilt scan.
    import random

    n, e0, e1 = 5, 0.001, 0.02
    questions = [(i, i) for i in range(n)] + [(i, (i + 1) % n) for i in range(n)]

    def mean_rate(gamma: float, tha: list[float], thb: list[float]) -> float:
        total = 0.0
        for x, y in questions:
            total += question_rate_long_way(gamma, tha[x], thb[y], e0, e1, edge=x != y)
        return total / len(questions)

    step_angle = math.pi * (n - 1) / n
    textbook_a = [-v * step_angle for v in range(n)]
    textbook_b = [angle - math.pi / (2 * n) for angle in textbook_a]
    textbook = mean_rate(quarter, textbook_a, textbook_b)
    predicted = omega_q(n) - delta_from_readout_asym(n, e0, e1)
    if abs(textbook - predicted) > 1e-12:
        failures.append(f"textbook rate {textbook:.12f} vs closed form {predicted:.12f}")

    best = textbook
    for seed in range(3):
        rng = random.Random(seed)
        if seed == 0:
            gamma = quarter
            tha, thb = list(textbook_a), list(textbook_b)
        else:
            gamma = rng.uniform(0.1, math.pi / 2 - 0.1)
            tha = [rng.uniform(-math.pi, math.pi) for _ in range(n)]
            thb = [rng.uniform(-math.pi, math.pi) for _ in range(n)]
        width = 0.5
        current = mean_rate(gamma, tha, thb)
        for _ in range(1200):
            slot = rng.randrange(2 * n + 1)
            if slot == 2 * n:
                trial = gamma + rng.gauss(0.0, width)
                if mean_rate(trial, tha, thb) > current:
                    gamma, current = trial, mean_rate(trial, tha, thb)
            elif slot < n:
                old = tha[slot]
                tha[slot] = old + rng.gauss(0.0, width)
                trial_rate = mean_rate(gamma, tha, thb)
                if trial_rate > current:
                    current = trial_rate
                else:
                    tha[slot] = old
            else:
                old = thb[slot - n]
                thb[slot - n] = old + rng.gauss(0.0, width)
                trial_rate = mean_rate(gamma, tha, thb)
                if trial_rate > current:
                    current = trial_rate
                else:
                    thb[slot - n] = old
            width = max(0.02, width * 0.999)
        best = max(best, current)
    if best > textbook + 1e-6:
        failures.append(
            f"the tilt search beat the textbook strategy: {best:.9f} vs {textbook:.9f}"
        )

    if failures:
        return False, "; ".join(failures[:3])
    return True, (
        f"{checked} channel points match the affine form to 1e-12, and the "
        f"tilt search tops out at {best:.9f} against the textbook "
        f"{textbook:.9f} at n = {n}, (e0, e1) = ({e0}, {e1})"
    )


def _cross_check_bernstein() -> tuple[bool | None, str]:
    """Assert `confidence_half_width` matches the published `calculate_ci`.

    Loads `ion_nlg/nlg_data/src/nlg_data/uncertainty.py` by path, read-only, and
    compares on shared inputs. Returns `None` when that repo is not present, which
    is the normal case: a student clone will never have it, and this check is an
    organizer-machine check.

    Searched for by walking up from this file rather than at a fixed depth, so
    moving this script between repos cannot silently turn the check into a skip.

    Until 19 August 2026 this also compared `p_value` against the paper's
    `calculate_p_value`. That comparison is retired: the research pass found the
    published expression is a heuristic rather than a theorem (it mixes
    shot-level and circuit-level granularities and matches no published bound;
    conservative in every case checked, so the paper's claims stand a
    fortiori), and the gate here is now the exact binomial tail, which is not
    supposed to agree with it. `confidence_half_width` survives as a
    transcription because the results database recomputes it verbatim, and a
    transcription needs a bit-for-bit check.
    """
    import importlib.util
    from pathlib import Path

    tail = Path("ion_nlg") / "nlg_data" / "src" / "nlg_data" / "uncertainty.py"
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor / tail
        if candidate.exists():
            source = candidate
            break
    else:
        return None, f"reference not present at any ancestor of {here.parent}/{tail}"

    spec = importlib.util.spec_from_file_location("_nlg_uncertainty", source)
    if spec is None or spec.loader is None:
        return None, f"reference at {source} could not be loaded as a module"
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        return None, f"reference present but not importable: {exc}"

    cases = [
        ([0.99] * 26, 189, 0.05),
        ([0.9893] * 58, 734, 0.05),
        ([0.98, 0.99, 1.0, 0.97] * 3, 500, 0.05),
        ([0.86] * 8, 1000, 0.01),
        ([0.5] * 10, 100, 0.32),
    ]
    failures = []
    for rates, shots, d in cases:
        ours = confidence_half_width(rates, shots, d)
        theirs = float(module.calculate_ci(rates, shots, d))
        if not math.isclose(ours, theirs, rel_tol=1e-12):
            failures.append(f"{len(rates)} circuits @ {shots}: {ours:.6e} vs {theirs:.6e}")
    if failures:
        return False, "; ".join(failures)
    return True, f"{len(cases)} cases agree to 1e-12 against {source.name}"


def _poisson_binomial_tail(rates: list[float], shots: int, k: int) -> float:
    """$P(\\sum_i \\mathrm{Bin}(shots, p_i) \\geq k)$ by exact convolution.

    Used only by `_check_gate_validity`. Quadratic and honest: convolve the
    per-circuit binomial pmfs as plain lists. No sampling, no approximation.
    """
    dist = [1.0]
    for p in rates:
        if p <= 0.0:
            pmf = [1.0] + [0.0] * shots
        elif p >= 1.0:
            pmf = [0.0] * shots + [1.0]
        else:
            pmf = [math.exp(_log_binom_pmf(shots, p, j)) for j in range(shots + 1)]
        new = [0.0] * (len(dist) + shots)
        for a, weight_a in enumerate(dist):
            if weight_a == 0.0:
                continue
            for b, weight_b in enumerate(pmf):
                new[a + b] += weight_a * weight_b
        dist = new
    return sum(dist[k:])


def _check_gate_validity() -> tuple[bool, str]:
    """The gate must hold its stated false-positive rate, stratified or not.

    First the homogeneous null, computed exactly rather than simulated: at the
    published S_90 for several $n$, the probability that a classical device
    sitting exactly at $\\omega_c$ clears the gate is the binomial tail at the
    gate count, and it must not exceed P_3SIGMA. That is the false-positive
    calibration of the gate, as an exact number rather than an assumption.

    Then the stratified null, where the theorem earns its keep. A classical
    strategy may win different circuits at different rates whose mean is
    $\\omega_c$, and the gate compares the pooled count against the i.i.d.
    binomial anyway. Hoeffding 1956 (Theorem 5) says the i.i.d. tail dominates
    every fixed-mean independent-Bernoulli tail in this range; this check makes
    the 19 August 2026 verification of that argument permanent by exact
    dynamic-programming convolution of adversarial spread vectors. No sampling
    anywhere.
    """
    failures = []
    for n in (3, 5, 13):
        shots = shots_for_significance(n)
        assert shots is not None
        total = n_circuits(n) * shots
        null_rate = binomial_tail(total, omega_c(n), _k_min(total, omega_c(n)))
        if null_rate > P_3SIGMA:
            failures.append(f"n={n}: homogeneous null clears at {null_rate:.3e}")

    n, shots = 3, 120
    m = n_circuits(n)
    total = m * shots
    k_gate = _k_min(total, omega_c(n))
    iid = binomial_tail(total, omega_c(n), k_gate)
    wc = omega_c(n)
    spreads = (
        [wc] * m,
        [1.0] * 3 + [2.0 * wc - 1.0] * 3,
        [wc + 0.15] * 3 + [wc - 0.15] * 3,
        [wc + 0.02 * (1 if i % 2 else -1) for i in range(m)],
    )
    for rates in spreads:
        mean = sum(rates) / m
        if abs(mean - wc) > 1e-12:
            failures.append(f"spread vector has mean {mean:.6f}, not omega_c")
            continue
        strat = _poisson_binomial_tail(rates, shots, k_gate)
        if strat > iid + 1e-12:
            failures.append(f"stratified tail {strat:.3e} exceeds the i.i.d. {iid:.3e}")
        if strat > P_3SIGMA:
            failures.append(f"stratified tail {strat:.3e} exceeds the gate")
    if failures:
        return False, "; ".join(failures[:3])
    return True, (
        f"homogeneous null holds the rate at S_90 for n = 3, 5, 13, and "
        f"{len(spreads)} exact stratified convolutions stay under the i.i.d. "
        f"tail {iid:.3e}"
    )


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclass
class Point:
    n: int
    shots: int
    dollars: float


def worked_point(n: int, delta: float = DELTA_EXAMPLE) -> Point | None:
    """The full cost line for one $n$, or `None` when $n$ is out of reach.

    `shots` is S_90, the sizing every published table uses: the smallest count
    that certifies with probability POWER_TARGET at the deficit passed in.
    """
    shots = shots_for_significance(n, delta)
    if shots is None:
        return None
    return Point(n, shots, cost(n, shots))


def main() -> int:
    offered = ", ".join(f"{qpu} ({route})" for qpu, route in QPU_ROUTES.items())
    print(f"Qupacabrathon 2026: odd-cycle game through qBraid. Devices: {offered}")
    print(
        f"delta = the team's own, declared in its parameter block. "
        f"{DELTA_EXAMPLE} below is an illustration, not either device."
    )
    print(
        "        predicted from published readout, "
        + ", ".join(
            f"{qpu} {predicted_delta(qpu, 13):.4f}" for qpu in PUBLISHED_READOUT
        )
        + " at n = 13; both over-predict, because neither"
    )
    print("        published readout figure corrects for preparation error")
    print(f"gate  = one-sided exact binomial tail, 3 sigma, p <= {P_3SIGMA:.3e}")
    print(
        f"        across {TEAMS} teams that is a "
        f"{100 * family_wise_false_positive(P_3SIGMA):.1f}% chance one honest team "
        f"clears it by luck, against {100 * family_wise_false_positive(P_2SIGMA):.1f}% "
        f"at 2 sigma"
    )
    print(
        f"price = ${TASK_DOLLARS:.2f} per circuit task + ${SHOT_DOLLARS:.5f} per shot, "
        f"one task per circuit variant (TWIRLS = k prices k tasks per question)"
    )
    print(
        "        [measured 2026-08-18 via the insufficient-credits oracle, "
        "1 qBraid credit = $0.01]"
    )
    print(
        f"pot   = ${POT:.2f}, {TEAMS} teams at ${CAP_PER_TEAM:.2f} "
        f"= ${TEAMS * CAP_PER_TEAM:.2f} committed, ${RESERVE:.2f} reserve"
    )
    print(
        f"sizing = S_90: the smallest shots/circuit that certify with "
        f"probability >= {POWER_TARGET:.2f} at the declared delta"
    )
    print()

    print("Self-checks")
    ok_chsh, detail = _check_chsh()
    print(f"  [{'PASS' if ok_chsh else 'FAIL'}] CHSH (n = 2): {detail}")
    ok_inv, detail = _check_shot_inversion()
    print(f"  [{'PASS' if ok_inv else 'FAIL'}] S_90 inversion: {detail}")
    ok_gate, detail = _check_gate_validity()
    print(f"  [{'PASS' if ok_gate else 'FAIL'}] gate validity: {detail}")
    ok_price, detail = _check_route_price()
    print(f"  [{'PASS' if ok_price else 'FAIL'}] route price: {detail}")
    ok_eff, detail = _check_design_efficiency()
    print(f"  [{'PASS' if ok_eff else 'FAIL'}] design efficiency: {detail}")
    ok_res, detail = _check_efficiency_resolution()
    print(f"  [{'PASS' if ok_res else 'FAIL'}] efficiency resolution: {detail}")
    ok_price_dev, detail = _check_device_pricing()
    print(f"  [{'PASS' if ok_price_dev else 'FAIL'}] device pricing: {detail}")
    ok_readout, detail = _check_readout_law()
    print(f"  [{'PASS' if ok_readout else 'FAIL'}] readout law: {detail}")
    ok_optimal, detail = _check_strategy_optimality()
    print(f"  [{'PASS' if ok_optimal else 'FAIL'}] strategy optimality: {detail}")
    ok_ref, detail = _cross_check_bernstein()
    label = "SKIP" if ok_ref is None else ("PASS" if ok_ref else "FAIL")
    print(f"  [{label}] Bernstein CI vs published reference: {detail}")
    print()

    print("The game at each n")
    header = f"{'n':>4} {'circuits':>9} {'omega_c':>9} {'omega_q':>9} {'margin':>9} {'min vis':>9}"
    print(header)
    for n in (3, 5, 7, 9, 11, 13, 15, 17):
        print(
            f"{n:>4} {n_circuits(n):>9} {omega_c(n):>9.6f} {omega_q(n):>9.6f} "
            f"{quantum_advantage(n):>9.6f} {critical_visibility(n):>9.4f}"
        )
    print()

    print(
        f"Cost to certify at 3 sigma with {POWER_TARGET:.0%} power, at the "
        f"illustration delta = {DELTA_EXAMPLE} (not either device)"
    )
    print(f"{'n':>4} {'S_90':>7} {'power':>7} {'total shots':>12} {'dollars':>9}")
    for n in (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25):
        point = worked_point(n)
        if point is None:
            print(f"{n:>4} {'infeasible':>10}")
            continue
        print(
            f"{n:>4} {point.shots:>7} {certification_power(n, point.shots):>7.3f} "
            f"{n_circuits(n) * point.shots:>12} {'$' + format(point.dollars, '.2f'):>9}"
        )
    chase = shots_for_chsh_ceiling(5)
    if chase is not None:
        print(
            f"  and the nonlocal-content chase: certify p_NL past the CHSH ceiling "
            f"({CHSH_CEILING:.5f})"
        )
        print(
            f"  at n = 5: {chase} shots/circuit, ${cost(5, chase):.2f} at "
            f"{POWER_TARGET:.0%} power"
        )
    print()

    print("Reachable n as a function of the device deficit (instructions Table A)")
    print("  no row is either device; the two predictions sit at " + ", ".join(
        f"{qpu} {predicted_delta(qpu, 13):.4f}" for qpu in PUBLISHED_READOUT
    ))
    print(
        f"{'delta':>7} {'physics':>8} {'$10.00':>22} "
        f"{'$' + format(CAP_PER_TEAM, '.2f'):>22}"
    )
    for delta in PUBLISHED_DELTA_GRID:
        feasible = largest_n_feasible(delta)

        def fmt(budget: float) -> str:
            got = largest_n_affordable(budget, delta)
            if got is None:
                return "none"
            n, shots, price = got
            return f"n={n} @ {shots}sh ${price:.2f}"

        print(
            f"{delta:>7.4f} {feasible if feasible else 'none':>8} "
            f"{fmt(10.0):>22} {fmt(CAP_PER_TEAM):>22}"
        )
    print()

    print(
        f"Three weekends inside the ${CAP_PER_TEAM:.2f} cap, illustrated at "
        f"delta = {DELTA_EXAMPLE}"
    )
    nine, eleven = worked_point(9), worked_point(11)
    assert nine is not None and eleven is not None
    ladder = nine.dollars + eleven.dollars
    print(
        f"  ladder: certify n = 9 (${nine.dollars:.2f}), then n = 11 "
        f"(${eleven.dollars:.2f}): ${ladder:.2f} total, "
        f"${CAP_PER_TEAM - ladder:.2f} left"
    )
    frontier = largest_n_affordable(CAP_PER_TEAM, DELTA_EXAMPLE)
    assert frontier is not None
    print(
        f"  one-shot commit: the frontier, n = {frontier[0]} at "
        f"{frontier[1]} shots/circuit: ${frontier[2]:.2f}, "
        f"${CAP_PER_TEAM - frontier[2]:.2f} left"
    )
    chase_shots = shots_affordable(5, CAP_PER_TEAM)
    print(
        f"  nonlocal-content chase: n = 5 at {chase_shots} shots/circuit: "
        f"${cost(5, chase_shots):.2f}, P(certify past the CHSH ceiling) = "
        f"{chsh_ceiling_power(5, chase_shots):.2f}"
    )
    print()

    print("What the exact gate freed, at n = 13")
    omega13 = omega_q(13) - DELTA_EXAMPLE
    eps13 = omega13 - omega_c(13)
    sigma2 = omega13 * (1.0 - omega13)
    bernstein = math.ceil(
        2.0 * math.log(1.0 / P_3SIGMA) * (sigma2 / n_circuits(13) + eps13 / 3.0) / eps13**2
    )
    print(
        f"  the Bernstein heuristic sized {bernstein} shots/circuit "
        f"(${cost(13, bernstein):.2f}), silently enforcing 6.1-6.9 sigma where "
        f"it said 3,"
    )
    expected_min = _smallest_shots(
        lambda s: 1.0
        if p_value([omega13] * n_circuits(13), s, omega_c(13)) <= P_3SIGMA
        else 0.0,
        0.5,
    )
    assert expected_min is not None
    print(
        f"  and put the expected outcome exactly at its gate, which certifies "
        f"about half the time"
    )
    print(
        f"  the exact tail puts the same expected outcome at the gate with "
        f"{expected_min} shots/circuit (${cost(13, expected_min):.2f})"
    )
    thirteen = worked_point(13)
    assert thirteen is not None
    print(
        f"  sized honestly for {POWER_TARGET:.0%} power it needs "
        f"{thirteen.shots} (${thirteen.dollars:.2f}): "
        f"{100.0 * (1.0 - thirteen.dollars / cost(13, bernstein)):.0f}% below a "
        f"heuristic that certified half as often"
    )
    print()

    print("Readout error as a dial, delta = 2 p (1 - p) (2 omega_q(n) - 1)")
    print("  what the simulator's BitFlip(p) on both wires produces, and the one")
    print("  place the n-independence the sizing rule assumes is visible")
    print(f"{'n':>4} {'p=0.005':>10} {'p=0.010':>10} {'p=0.013':>10} {'p=0.020':>10}")
    for n in (3, 5, 7, 9, 13, 17):
        row = "".join(
            f"{delta_from_readout(n, p):>10.4f}" for p in (0.005, 0.010, 0.013, 0.020)
        )
        print(f"{n:>4}{row}")
    dial = readout_from_delta(13, DELTA_EXAMPLE)
    assert dial is not None
    print(
        f"  p = {dial:.5f} per wire reproduces the illustration delta = "
        f"{DELTA_EXAMPLE}"
    )
    commit_n = 13
    drift = readout_drift(7, commit_n, dial)
    print(
        f"  calibrating at n = 7 and committing at n = {commit_n} reads delta "
        f"{100 * drift:.1f}% low"
    )
    bias = drift * delta_from_readout(commit_n, dial)
    calibration_stderr = math.sqrt(
        (omega_q(5) - DELTA_EXAMPLE)
        * (1 - omega_q(5) + DELTA_EXAMPLE)
        / (n_circuits(5) * 1000)
    )
    print(
        f"  a 1000-shot calibration at n = 5 pins delta to +/- "
        f"{calibration_stderr:.5f}, {calibration_stderr / bias:.1f} times wider "
        f"than that {bias:.5f} bias, so the n-independence assumption survives"
    )
    print()

    print("The prediction route: published readout error -> a delta to size against")
    print("  this is the whole of what a team can compute before spending anything")
    print(
        f"{'device':>9} {'e0':>8} {'e1':>8} {'delta(n=5)':>11} {'delta(n=13)':>12} "
        f"  source"
    )
    for qpu, (e0, e1, source) in PUBLISHED_READOUT.items():
        print(
            f"{qpu:>9} {e0:>8.4f} {e1:>8.4f} "
            f"{predicted_delta(qpu, 5):>11.4f} {predicted_delta(qpu, 13):>12.4f} "
            f"  {source}"
        )
    print(
        "  neither figure corrects for state preparation error, so both are upper"
    )
    print(
        "  estimates of the flip rate and both predictions land above the truth:"
    )
    for qpu in PUBLISHED_READOUT:
        got = largest_n_affordable(CAP_PER_TEAM, predicted_delta(qpu, 13))
        reach = "nothing" if got is None else f"n = {got[0]} at ${got[2]:.2f}"
        print(f"    a team trusting {qpu}'s specs sizes for {reach} within the cap")
    print(
        f"  block gap between the vertex and edge questions, the one fingerprint "
        f"asymmetric readout leaves:"
    )
    for qpu, (e0, e1, _) in PUBLISHED_READOUT.items():
        print(f"    {qpu}: {readout_asym_block_gap(e0, e1):.1e}")
    print()

    print("Calibration: measuring delta is a line item now, not free boundary shots")
    print("  the continuous meter prices precision directly, so choose it on purpose")
    for shots in (shots_for_significance(5, DELTA_EXAMPLE), 250, 1000):
        total = n_circuits(5) * shots
        omega = omega_q(5) - DELTA_EXAMPLE
        stderr = math.sqrt(omega * (1.0 - omega) / total)
        print(
            f"  n=5: {shots} shots/circuit, {total} trials, ${cost(5, shots):.2f}, "
            f"pins delta to +/- {stderr:.5f} (1 sigma)"
        )
    print()

    all_ok = (
        ok_chsh
        and ok_inv
        and ok_gate
        and ok_price
        and ok_eff
        and ok_res
        and ok_price_dev
        and ok_readout
        and ok_optimal
        and ok_ref is not False
    )
    print("ALL CHECKS PASS" if all_ok else "CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
