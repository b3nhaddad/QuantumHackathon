# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pennylane==0.45.1",
#     "qbraid==0.12.2",
#     "numpy==2.5.2",
# ]
# ///
"""Qupacabrathon 2026 submission script for the odd-cycle nonlocal game.

Copy this file to `submissions/<your-team>_run1.py`, fill in `build_circuits`,
set the parameter block, and open a pull request. An organizer checks your branch
out and runs it against the event key. Nobody touches hardware interactively, so
that one file is the whole submission. This template stays as it is: it is what
the next copy starts from.

    uv run --python 3.12 submissions/<your-team>_run1.py

The code you add here is small on purpose. `build_circuits` is the one function
your team fills in, and the parameter block above it is where the decisions that
differentiate teams land: which n, how many shots, whether to pay for twirled
variants, informed by what your run 1 measured. The experiment is the work and
this file is its instrument. Everything outside those two places is fixed so
that twelve submissions have a common surface a checker can validate and a
mentor can read, and editing it means your run is no longer the run that was
checked.

The script writes `counts.json` and `benchmark.json` into
`results/<your-team>_run<N>/`. That is exactly the shape the results database
expects, so an organizer files the result without transforming anything.

Before you open the pull request:

    uv run --python 3.12 scripts/submission_check.py submissions/<your-team>_run1.py

Every bound, shot count and price this script prints comes from
`scripts/game_numbers.py`. Nothing is restated here, so a number can never drift
between the rules and the run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy
import pennylane as qml

# `scripts/` is a sibling of this file's directory, whether the file sits in
# `challenge/` or in `submissions/`. Resolving it from `__file__` rather than from
# the working directory is what lets an organizer run this script from anywhere
# without editing a line of it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from game_numbers import (  # noqa: E402
    CAP_PER_TEAM,
    CHSH_CEILING,
    MINIMUM_SHOTS,
    P_3SIGMA,
    QPU_ROUTES,
    certified_p_nl,
    confidence_half_width,
    cost,
    n_circuits,
    omega_c,
    omega_q,
    p_value,
    rates_for_qpu,
)

# ==========================================================================
# PARAMETER BLOCK. This is what an organizer reads to know what a run costs.
# ==========================================================================

# Your team's chosen name, underscored with the number the organizers assigned
# you. The same identifier names your branch, your row in the spend ledger and
# your folder in the results database, so it has to match all three.
TEAM = "your_team_name_00"

# Which of your two hardware runs this is, 1 or 2.
RUN = 1

# Cycle size. Odd and at least 3. This is the competition axis: a better run
# shows up as a larger certified n, not as a third decimal place.
N = 13

# Shots per circuit. Every one of the 2n circuits gets exactly this many, and
# the verifier rejects a submission where they differ. Size it with
# `scripts/plan_check.py` before you submit, because both the certification gate
# and the bill are decided here. The published tables are S_90: the smallest
# count that certifies with 90% probability at the deficit you declared below.
# Buying the bare minimum instead puts the expected outcome exactly at the gate
# and fails about half the time.
SHOTS = 53

# Pauli-twirled circuit variants per question. 1 submits your circuit exactly
# as you wrote it. At k > 1, each question is submitted as k variants, each
# wrapping every CNOT in fresh random Pauli frames with the compensating pair
# after it, and each variant runs SHOTS / k shots, so SHOTS stays your total
# per question and has to divide by k. Every frame is the same unitary as your
# circuit, so the pooled counts are the same honest trials the certification
# counts. What twirling changes is the device's error, folding coherent error
# on the entangling gate into a stochastic channel. The price is real: one
# task fee per variant, so k twirls multiply the $0.30 task term by k while
# the shot term stays put. Whether it buys anything on this device is
# unmeasured, which is what makes it a bet rather than a default. See the
# Challenge Instructions, What you may change about the run.
TWIRLS = 1

# Which QPU this run goes to: "emerald" or "garnet", both IQM superconducting
# devices reached through qBraid. This is your choice and it is a real one, and
# it changes two separate things.
#
# It changes the price. Both devices charge $0.30 per circuit task; Emerald
# charges $0.0016 per shot and Garnet $0.00145, so a shot-heavy sweep is about
# 9% cheaper on Garnet and a task-heavy one costs the same on both. Every price
# this script prints is computed at the rate for the device named here.
#
# It changes the error rates, and so the shot count. The Background Notes carry
# a table of what each vendor publishes. A worse device needs more shots for
# the same certification, and the shots a certification needs grow like the
# inverse square of the margin, so a difference in error rates usually swamps a
# 10% difference in the per-shot rate.
#
# You name the device, never the route that reaches it. The organizer's
# environment maps this label to a route at execution time, which is what lets
# a device be swapped for its fallback without touching a submission.
QPU = "emerald"

# The device deficit you are sizing this run against: how far below the ideal
# quantum win rate you expect the hardware to land, omega = omega_q - delta.
#
# The event publishes no value for this. It is the number your experiment is
# designed on, and reaching a defensible one is most of the design work. Two
# routes. Predict it: the Background Notes derive the law that turns a readout
# error rate into a deficit, and carry the published error rates for both
# devices. Or measure it: a cheap sweep at low n returns a win rate, and
# omega_q(n) minus that win rate is delta, measured on the device you will
# commit on. Because delta does not depend on n, one cheap run sizes every
# later one.
#
# Nothing checks this against the device before you spend. `plan_check.py`
# checks only that your shot count and your budget are consistent with the
# number you declared. Declaring a deficit better than the truth means the gate
# passes here and the hardware certifies nothing; declaring one worse means you
# over-buy shots and pay for it on the design award. Both are priced, and both
# are your call.
DELTA = 0.015

# The one line that differs between a practice run and a hardware run, and it
# is not yours to set: leave it as it ships. An organizer swaps this single
# string to "qbraid" at execution time, after a mentor approved the pull
# request, which is why a script that runs by accident spends nothing and why
# you hold no key. On the organizer machine the "qbraid" route resolves your
# QPU label above through their environment, so no route string ever appears in
# this file and a device can be swapped for its fallback without touching a
# single submission. `check_environment` below stops a "qbraid" run whose
# environment carries no route or no credential before the first shot.
DEVICE = "default.qubit"

# Everything above is read once, before your code runs, and the run is executed
# from that snapshot. Reassigning any of it later changes nothing except that the
# script stops, because the parameters an organizer approved on the pull request
# have to be the parameters that spend the money.


# ==========================================================================
# THE PART YOUR TEAM WRITES
# ==========================================================================


def build_circuits(n: int, theta: float) -> list[Callable[[], None]]:
    # theta = pi / (4n)
    #
    # The optimal odd-cycle strategy uses:
    #
    #   Alice(x) = x * (pi - pi/n)
    #   Bob(y)   = y * (pi - pi/n) + pi/(2n)
    #
    # Since the template gives us theta:
    #
    #   pi/n    = 4 theta
    #   pi/(2n) = 2 theta

    step = math.pi - 4.0 * theta

    def alice_angle(x: int) -> float:
        return x * step

    def bob_angle(y: int) -> float:
        return y * step + 2.0 * theta

    def gates_for(x: int, y: int) -> Callable[[], None]:
        angle_a = alice_angle(x)
        angle_b = bob_angle(y)

        def circuit() -> None:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])

            qml.RY(angle_a, wires=0)
            qml.RY(angle_b, wires=1)

        return circuit

    return [
        gates_for(x, y)
        for x, y in question_order(n)
    ]

# ==========================================================================
# FIXED. Do not edit below this line.
# ==========================================================================


def question_order(n: int) -> list[tuple[int, int]]:
    """The 2n questions of one sweep, in the order everything downstream assumes.

    First the n same-vertex questions (i, i), then the n edges
    (i, (i + 1) mod n), each edge once in that single orientation. Taking each
    edge once rather than in both directions is what makes the classical bound
    1 - 1/(2n), so reversing an edge or adding its mirror moves the bound the
    run is scored against without raising an error anywhere.
    """
    same_vertex = [(i, i) for i in range(n)]
    edges = [(i, (i + 1) % n) for i in range(n)]
    return same_vertex + edges


def question_key(x: int, y: int) -> str:
    """Question key `"<x>|<y>"`, decimal and unpadded, as the database reads it."""
    return f"{x}|{y}"


def answer_key(a: int, b: int) -> str:
    """Answer key `"<a>:<b>"`, one bit per player, first player leftmost.

    Fixed width matters even at one bit per player: without it two spellings of
    the same answer can split one bin, and the recomputed win rate stops
    matching the measured data.
    """
    return f"{a}:{b}"


def is_win(x: int, y: int, a: int, b: int) -> bool:
    """The odd-cycle win condition.

    Same vertex asked twice, the players have to agree on its color. Adjacent
    vertices, they have to disagree. No two-coloring of an odd cycle satisfies
    every edge, which is the whole reason the game separates classical from
    quantum.
    """
    if x == y:
        return a == b
    return a != b


def strategy_angle(n: int) -> float:
    """The angle whose cosine squared is the quantum bound, passed to `build_circuits`.

    Derived from `omega_q` rather than written down, so this line cannot drift
    from `scripts/game_numbers.py`.
    """
    return math.acos(math.sqrt(omega_q(n)))


def tally_answers(samples: object, shots: int) -> dict[str, int]:
    """Turn one circuit's raw samples into the four answer counts.

    Raises rather than repairing anything. The counting bound behind the
    certification is a statement about honest Bernoulli trials, so a circuit
    that came back with the wrong number of shots or an answer outside {0, 1} is
    a failed run to report, never a partial result to salvage.
    """
    rows = numpy.asarray(samples).reshape(-1, 2)
    if rows.shape[0] != shots:
        raise RuntimeError(f"device returned {rows.shape[0]} shots, expected {shots}")

    counts = {answer_key(a, b): 0 for a in (0, 1) for b in (0, 1)}
    for row in rows:
        a, b = int(row[0]), int(row[1])
        if a not in (0, 1) or b not in (0, 1):
            raise RuntimeError(f"answer ({a}, {b}) is not a pair of bits")
        counts[answer_key(a, b)] += 1

    # The verifier checks this for every circuit, so check it here where the
    # failure is still attributable to one circuit.
    if sum(counts.values()) != shots:
        raise RuntimeError(f"counts sum to {sum(counts.values())}, expected {shots}")
    return counts


def question_win_rate(x: int, y: int, counts: dict[str, int], shots: int) -> float:
    """Fraction of this question's shots that won it."""
    wins = sum(
        counts[answer_key(a, b)]
        for a in (0, 1)
        for b in (0, 1)
        if is_win(x, y, a, b)
    )
    return wins / shots


def hardware_route() -> str | None:
    """The qBraid device the executing machine routes to, if any.

    No credential passes through here. qBraid Runtime reads QBRAID_API_KEY out
    of the environment itself, so a key is never an argument, never a literal
    and never in the output. What this reads is the QPU the organizer routes
    to, which is an organizer decision and changes between the announced device
    and its fallback. That is why no hardware name appears in this file, and
    why DEVICE = "qbraid" is a declaration of intent rather than a destination.
    """
    route = os.environ.get("QBRAID_DEVICE", "").strip()
    return route or None


def check_environment(device_string: str, route: str | None) -> None:
    """Fail before the first shot when a hardware run has nowhere to go.

    Only presence is checked. Reading the credential's value into this script
    would put a key one traceback away from the output, and the runtime reads
    it directly anyway.

    Gated on the device string: DEVICE is the one line that differs between
    practice and hardware, so it is the one line this decision may rest on. A
    hardware DEVICE with nothing configured fails here rather than inside the
    runtime.
    """
    if device_string != "qbraid":
        return
    if route is None:
        raise RuntimeError(
            "DEVICE is 'qbraid' but the environment names no QPU: the executing "
            "machine sets QBRAID_DEVICE. That is the organizer's machine, so if "
            "you are seeing this on your laptop, run on 'default.qubit' and "
            "submit the pull request instead."
        )
    if not (os.environ.get("QBRAID_API_KEY") or os.environ.get("QBRAID_ACCESS_TOKEN")):
        raise RuntimeError(
            "hardware route requested but the environment carries no credential: "
            "set QBRAID_API_KEY (or QBRAID_ACCESS_TOKEN)"
        )


def check_parameters() -> None:
    """Reject a parameter block that cannot produce a valid submission."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{3,50}", TEAM):
        raise ValueError(
            f"TEAM = {TEAM!r} is not a valid identifier. Use your team name "
            "underscored with the number the organizers assigned you, letters, "
            "digits, underscores and hyphens only."
        )
    if RUN not in (1, 2):
        raise ValueError(
            f"RUN = {RUN} is not 1 or 2. Each team gets two hardware runs."
        )
    if not isinstance(N, int) or N < 3 or N % 2 == 0:
        raise ValueError(f"N = {N} is not an odd cycle. Pick an odd n at least 3.")
    # The results database declares n at most 99 for this game, so a larger run
    # would execute, bill, and then be unfileable.
    if N > 99:
        raise ValueError(
            f"N = {N} is above 99, which the results database cannot file."
        )
    if not isinstance(SHOTS, int) or SHOTS < MINIMUM_SHOTS:
        raise ValueError(
            f"SHOTS = {SHOTS} is below the {MINIMUM_SHOTS} floor the event "
            "enforces."
        )
    if not isinstance(TWIRLS, int) or isinstance(TWIRLS, bool) or not 1 <= TWIRLS <= 16:
        raise ValueError(
            f"TWIRLS = {TWIRLS} is not an integer between 1 and 16. Each twirl "
            "is a separately billed task per question, so the count is bounded."
        )
    if SHOTS % TWIRLS != 0:
        raise ValueError(
            f"SHOTS = {SHOTS} does not split into TWIRLS = {TWIRLS} equal "
            "variants. Every variant gets the same honest trials, so pick "
            "SHOTS as a multiple of TWIRLS. The published S_90 tables are "
            "sized at TWIRLS = 1; round S up to the next multiple."
        )
    if SHOTS // TWIRLS < MINIMUM_SHOTS:
        raise ValueError(
            f"SHOTS = {SHOTS} at TWIRLS = {TWIRLS} is {SHOTS // TWIRLS} shots "
            f"per variant, below the {MINIMUM_SHOTS} floor the device enforces "
            "per task."
        )
    if QPU not in QPU_ROUTES:
        raise ValueError(
            f"QPU = {QPU!r} is not a device this event offers. Pick one of "
            f"{sorted(QPU_ROUTES)}. Name the device, never the route: a route "
            "string in a submission is rejected by the gate."
        )
    # No check here compares DELTA against the device, because nothing in this
    # repository knows what the device does. What is checked is that it is a
    # deficit at all, and that the plan built on it is internally consistent.
    if isinstance(DELTA, bool) or not isinstance(DELTA, (int, float)):
        raise ValueError(f"DELTA = {DELTA!r} is not a number.")
    if not 0.0 <= DELTA < 0.5:
        raise ValueError(
            f"DELTA = {DELTA} is not a device deficit in [0, 0.5). It is "
            "omega_q - omega, the shortfall you expect, so it is a small "
            "positive number."
        )
    # Twelve teams spend from one pot, so an arithmetic mistake here costs every
    # other team. This is the same ceiling plan_check.py applies, checked again
    # against the parameters that are actually about to execute.
    planned = cost(N, SHOTS, TWIRLS, QPU)
    if planned > CAP_PER_TEAM:
        raise ValueError(
            f"N = {N} at {SHOTS} shots and {TWIRLS} twirl(s) on {QPU} costs "
            f"${planned:.2f}, above the ${CAP_PER_TEAM:.2f} per-team cap. "
            "Size it with scripts/plan_check.py."
        )


@qml.transform
def fold_ry_angles(
    tape: qml.tape.QuantumScript,
) -> tuple[tuple[qml.tape.QuantumScript, ...], Callable[[list], object]]:
    """Rewrite every `RY(theta)` as `RY(r)` followed by `Y`, where `theta = m*pi + r`.

    Measured on `iqm:garnet` on 2026-08-18: the backend folds an `RY` angle into
    roughly `(-pi/2, pi/2]` below the QASM this script emits. That is not a
    harmless wrap. `RY(theta + pi) = -i * Y * RY(theta)`, so dropping a pi flips
    the qubit, and a strategy whose questions differ by a step of `pi - pi/n`
    collapses onto a single question. A `C_7` sweep whose angles were correct in
    the emitted QASM came back at a win rate of 0.4996, chance, against 0.9858
    for the same file on the simulator. The two evidence circuits: `RY(-pi)` on
    one half of a Bell pair returned `P(same) = 0.9788` where the exact answer is
    0, and the same rotation written as `Y` returned 0.0318. The 2026-08-19
    Emerald session ran this exact rewrite and certified at `n = 5`, so on the
    event device, which shares the IQM compiler stack, the workaround is
    validated empirically rather than assumed to transfer.

    So the pi is carried by a Pauli, which no angle folding can touch, and only
    the remainder is left in an angle. `math.remainder` returns `r` in
    `[-pi/2, pi/2]` by construction, which is inside the range the backend keeps.

    The rewrite is exact rather than approximate. It agrees with `RY(theta)` up to
    a global phase, verified to 1.7e-16 in the Bell-pair probabilities across the
    angles this game generates, so the simulator cannot tell the two apart and the
    number the gate prints is the number the run is trying to reproduce.

    This sits below the FIXED line because it is a property of the device rather
    than of any team's strategy. A team writes `qml.RY` in `build_circuits` and
    this makes it survive the trip.
    """
    operations = []
    for op in tape.operations:
        if isinstance(op, qml.RY):
            remainder = math.remainder(float(op.parameters[0]), math.pi)
            half_turns = round((float(op.parameters[0]) - remainder) / math.pi)
            operations.append(qml.RY(remainder, wires=op.wires))
            # Y squares to the identity up to a phase, so only the parity counts.
            if half_turns % 2:
                operations.append(qml.PauliY(wires=op.wires))
        else:
            operations.append(op)

    folded = tape.copy(operations=operations)
    return (folded,), lambda results: results[0]


# The sixteen Pauli frames a twirl draws from, and where CNOT carries each one.
# `_CNOT_CONJUGATE[(c, t)]` is the pair `(c', t')` with
# `(P_c' x P_t') CNOT (P_c x P_t) = CNOT` up to a global phase, so inserting a
# frame before the CNOT and its image after it leaves every variant exactly
# unitarily equivalent to the circuit the team wrote. `_check_twirl_table`
# verifies that identity numerically at import, so the table cannot rot
# silently.
_TWIRL_FRAMES: tuple[tuple[str, str], ...] = tuple(
    (control, target) for control in "IXYZ" for target in "IXYZ"
)
_CNOT_CONJUGATE: dict[tuple[str, str], tuple[str, str]] = {
    ("I", "I"): ("I", "I"),
    ("I", "X"): ("I", "X"),
    ("I", "Y"): ("Z", "Y"),
    ("I", "Z"): ("Z", "Z"),
    ("X", "I"): ("X", "X"),
    ("X", "X"): ("X", "I"),
    ("X", "Y"): ("Y", "Z"),
    ("X", "Z"): ("Y", "Y"),
    ("Y", "I"): ("Y", "X"),
    ("Y", "X"): ("Y", "I"),
    ("Y", "Y"): ("X", "Z"),
    ("Y", "Z"): ("X", "Y"),
    ("Z", "I"): ("Z", "I"),
    ("Z", "X"): ("Z", "X"),
    ("Z", "Y"): ("I", "Y"),
    ("Z", "Z"): ("I", "Z"),
}
_PAULI_OPS = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}


def _check_twirl_table() -> None:
    """Assert every frame's compensation restores CNOT, up to a global phase.

    Sixteen 4 x 4 matrix products against the CNOT matrix. Runs once at
    import, which is what makes the table above a checked fact rather than a
    transcription.
    """
    single = {
        "I": numpy.eye(2),
        "X": numpy.array([[0, 1], [1, 0]]),
        "Y": numpy.array([[0, -1j], [1j, 0]]),
        "Z": numpy.array([[1, 0], [0, -1]]),
    }
    cnot = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    for frame in _TWIRL_FRAMES:
        after = _CNOT_CONJUGATE[frame]
        before = numpy.kron(single[frame[0]], single[frame[1]])
        compensation = numpy.kron(single[after[0]], single[after[1]])
        product = compensation @ cnot @ before
        # Divide out the global phase read off the largest entry, then compare.
        index = numpy.unravel_index(numpy.argmax(numpy.abs(product)), product.shape)
        phase = product[index] / abs(product[index])
        if not numpy.allclose(product / phase, cnot, atol=1e-12):
            raise AssertionError(f"twirl frame {frame} does not restore CNOT")


_check_twirl_table()


@qml.transform
def pauli_twirl(
    tape: qml.tape.QuantumScript, rng: random.Random
) -> tuple[tuple[qml.tape.QuantumScript, ...], Callable[[list], object]]:
    """Wrap every CNOT in a fresh random Pauli frame, compensated after it.

    Each variant draws one of the sixteen frames per CNOT from `rng` and
    inserts the frame before the gate and its `_CNOT_CONJUGATE` image after
    it, so the variant is the team's circuit exactly, up to a global phase.
    On a perfect device nothing changes; on a device whose CNOT carries a
    coherent error, averaging over frames folds that error into a stochastic
    channel, which is the mechanism twirling is bought for.

    This sits below the FIXED line for the same reason `fold_ry_angles` does:
    the frames are not a strategy. The team's decision is the TWIRLS
    parameter, which is a priced bet, and the frames themselves are drawn
    fresh per variant by `run_sweep` with a seed named after the question and
    the variant, so the run an organizer executes is reproducible.
    """
    operations = []
    for op in tape.operations:
        if isinstance(op, qml.CNOT):
            control, target = op.wires
            frame = _TWIRL_FRAMES[rng.randrange(len(_TWIRL_FRAMES))]
            after = _CNOT_CONJUGATE[frame]
            for label, wire in zip(frame, (control, target), strict=True):
                if label != "I":
                    operations.append(_PAULI_OPS[label](wires=wire))
            operations.append(qml.CNOT(wires=[control, target]))
            for label, wire in zip(after, (control, target), strict=True):
                if label != "I":
                    operations.append(_PAULI_OPS[label](wires=wire))
        else:
            operations.append(op)

    twirled = tape.copy(operations=operations)
    return (twirled,), lambda results: results[0]


def _variant_rng(index: int, variant: int) -> random.Random:
    """The seeded frame source for one variant of one question.

    Named by the question's fixed-order index and the variant number, so the
    frames a run drew can be reproduced from the submission alone.
    """
    return random.Random(f"pauli-twirl-{index}-{variant}")


def measured_circuit(
    device: object,
    gates: Callable[[], None],
    shots: int,
    twirl_rng: random.Random | None = None,
) -> Callable[[], object]:
    """Attach the measurement to one team-written circuit.

    Every circuit is measured the same way, on the same two wires, because the
    win rate is only comparable across questions if the answers are read the same
    way at every question. Player A's bit comes first in the sample, at every
    question.

    `shots` is passed in rather than read from the parameter block, so the shot
    count that executes is the one that was priced and approved. `twirl_rng`,
    when given, twirls the circuit's CNOTs before the angle rewrite; at
    TWIRLS = 1 it stays `None` and the circuit runs exactly as written.
    """

    @qml.qnode(device)
    def circuit() -> object:
        gates()
        return qml.sample(wires=[0, 1])

    transformed = circuit if twirl_rng is None else pauli_twirl(circuit, twirl_rng)
    return qml.set_shots(fold_ry_angles(transformed), shots=shots)


def circuit_qasm(
    gates: Callable[[], None], shots: int, twirl_rng: random.Random | None = None
) -> str:
    """One measured circuit as an OpenQASM 2 string, for the hardware route.

    The same QNode the simulator path runs, exported through `qml.to_openqasm`
    with the `pauli_twirl` and `fold_ry_angles` rewrites already applied, so
    the QASM the device receives is gate-for-gate the circuit the gate
    simulated.
    """

    @qml.qnode(qml.device("default.qubit", wires=2, shots=shots))
    def circuit() -> object:
        gates()
        return qml.sample(wires=[0, 1])

    transformed = circuit if twirl_rng is None else pauli_twirl(circuit, twirl_rng)
    return qml.to_openqasm(fold_ry_angles(transformed))()


def tally_bitstring_counts(raw: dict[str, int], shots: int) -> dict[str, int]:
    """Turn a hardware counts dict, keyed by bitstring, into the four answer counts.

    The runtime returns `{"01": 137, ...}` with wire 0 leftmost. [UNVERIFIED:
    the bit order is confirmed at the dry run. It cannot move the win rate,
    because both win conditions of this game are symmetric under swapping the
    two answer bits; what it could mislabel is which player answered what.]

    Raises rather than repairing, for the reason `tally_answers` gives.
    """
    counts = {answer_key(a, b): 0 for a in (0, 1) for b in (0, 1)}
    for bits, count in raw.items():
        bits = bits.strip()
        if len(bits) != 2 or any(bit not in "01" for bit in bits):
            raise RuntimeError(f"answer key {bits!r} is not a pair of bits")
        counts[answer_key(int(bits[0]), int(bits[1]))] += int(count)
    if sum(counts.values()) != shots:
        raise RuntimeError(f"counts sum to {sum(counts.values())}, expected {shots}")
    return counts


def run_sweep(
    questions: list[tuple[int, int]],
    circuits: list[Callable[[], None]],
    device_string: str,
    shots: int,
    route: str | None,
    twirls: int = 1,
) -> dict[str, dict[str, int]]:
    """Execute every circuit at `shots` total shots and return the raw counts.

    One device, one task per circuit variant, which is the unit the price is
    quoted in. At `twirls` = 1 each question is one task; at k > 1 it is k
    twirled variants of `shots // twirls` shots each, pooled per question, so
    every question still contributes exactly `shots` honest trials and the
    verifier's per-circuit sum is unchanged. Shots travel with each variant
    rather than sitting on a shared device, so no circuit can be allocated
    more or fewer than the parameter block declares.

    On the hardware route the circuits are submitted in a randomised order, so
    that a drift in the device over the sweep cannot correlate with the question
    index; the counts are keyed by question, so nothing downstream changes. The
    simulator path keeps the fixed order, which is what makes a check
    reproducible. Twirl frames are seeded by question and variant on both
    paths, for the same reason.

    The device string, the shot count and the twirl count are arguments rather
    than globals, because the sweep between them is the whole bill. A run has
    to spend what was approved even if something further down the file says
    otherwise.

    Nothing here reprocesses what the device returned. Pooling a question's
    variants is addition of counts of identical circuits, up to a global
    phase, before any analysis. Post-execution mitigation is disallowed, and
    feeding a fit or a corrected count into a bound that counts trials would
    make the bound meaningless.
    """
    if twirls < 1 or shots % twirls != 0:
        raise ValueError(f"shots = {shots} does not split into {twirls} variants")
    per_variant = shots // twirls
    counts: dict[str, dict[str, int]] = {}

    if route is not None:
        from qbraid.runtime import QbraidProvider

        device = QbraidProvider().get_device(route)
        order = list(range(len(questions)))
        random.shuffle(order)
        for position, index in enumerate(order, 1):
            x, y = questions[index]
            table = {answer_key(a, b): 0 for a in (0, 1) for b in (0, 1)}
            for variant in range(twirls):
                rng = _variant_rng(index, variant) if twirls > 1 else None
                job = device.run(
                    circuit_qasm(circuits[index], per_variant, twirl_rng=rng),
                    shots=per_variant,
                )
                raw = job.result().data.get_counts()
                part = tally_bitstring_counts(raw, per_variant)
                for answer, count in part.items():
                    table[answer] += count
            key = question_key(x, y)
            counts[key] = table
            rate = question_win_rate(x, y, table, shots)
            print(f"  [{position:>3}/{len(questions)}] {key:<9} win rate {rate:.4f}")
        # Reassemble the fixed question order the verifier and database expect.
        return {
            question_key(x, y): counts[question_key(x, y)] for x, y in questions
        }

    device = qml.device(device_string, wires=2)
    for index, ((x, y), gates) in enumerate(zip(questions, circuits, strict=True), 1):
        table = {answer_key(a, b): 0 for a in (0, 1) for b in (0, 1)}
        for variant in range(twirls):
            rng = _variant_rng(index - 1, variant) if twirls > 1 else None
            samples = measured_circuit(device, gates, per_variant, twirl_rng=rng)()
            part = tally_answers(samples, per_variant)
            for answer, count in part.items():
                table[answer] += count
        key = question_key(x, y)
        counts[key] = table
        rate = question_win_rate(x, y, table, shots)
        print(f"  [{index:>3}/{len(questions)}] {key:<9} win rate {rate:.4f}")

    return counts


def counts_document(counts: dict[str, dict[str, int]]) -> dict[str, object]:
    """The counts file, in the encoding the results database verifies against."""
    return {"schemaVersion": 1, "counts": counts}


def benchmark_document(
    win_rate: float,
    half_width: float,
    p: float,
    p_nl: float,
    device_label: str,
    team: str,
    run: int,
    n: int,
    shots: int,
    twirls: int,
) -> dict[str, object]:
    """The benchmark file. `nonlocalGame` is the claim that gets recomputed.

    The database recalculates the win rate from the counts and compares it with
    what is claimed here, so these two files are checked against each other
    rather than taken on trust.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )
    certified = "certified" if p <= P_3SIGMA else "not certified"
    return {
        "algorithmName": f"Odd Cycle C{n} (Nonlocal Game)",
        "device": device_label,
        "metricName": "Win Rate",
        "metricValue": win_rate,
        "uncertainty": half_width,
        "description": (
            f"Odd cycle nonlocal game on C_{n}, {n_circuits(n)} question circuits at "
            f"{shots} shots each, run {run} of the Qupacabrathon 2026 hackathon."
        ),
        "team": [team],
        "timestamp": now,
        "experimentDate": now,
        "quantumSpecific": {
            "qubitCount": 2,
            "shots": n_circuits(n) * shots,
        },
        "nonlocalGame": {
            "game": "odd-cycle",
            "params": {"n": n},
            "winRate": win_rate,
            "shotsPerCircuit": shots,
            "countsFile": "counts.json",
            "uncertainty": half_width,
            "uncertaintyDefinition": (
                f"95% confidence half-width (d = 0.05) over the {n_circuits(n)} "
                f"per-question win rates at {shots} shots each, the Bernstein "
                "interval calculate_ci the results database recomputes"
            ),
            "allowVariableShots": False,
            "eventTeam": team,
            "provenance": {
                "event": "Qupacabrathon 2026",
                "run": run,
                "device": device_label,
            },
        },
        "notes": (
            f"Classical bound omega_c = {omega_c(n)}. Quantum bound "
            f"omega_q = {omega_q(n)}. One-sided exact binomial tail "
            f"p = {p:.6e} against the 3 sigma threshold {P_3SIGMA:.6e}, so this "
            f"run is {certified} at 3 sigma. Certified nonlocal content "
            f"p_NL = {p_nl:.6f} (exact binomial lower bound at the same "
            f"threshold; the CHSH ceiling is {CHSH_CEILING:.5f}). "
            + (
                f"Each question ran as {twirls} Pauli-twirled variants of "
                f"{shots // twirls} shots each, every variant unitarily "
                "equivalent to the submitted circuit, pooled per question. "
                if twirls > 1
                else ""
            )
            + "The counts are what the device returned: no readout "
            "correction, no extrapolation and no postselection was applied "
            "to them."
        ),
    }


def write_json(path: Path, document: dict[str, object]) -> str:
    """Write one output file, and return the SHA-256 of exactly what was written.

    The digest is printed. An organizer executes this script and captures its
    output, so the log carries an independent record of the counts the device
    produced, and a file that changed afterwards no longer matches it.
    """
    text = json.dumps(document, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    check_parameters()

    # The parameter block is read once, here, before any team-written code runs,
    # and everything below uses these locals. `build_circuits` is arbitrary Python
    # executing on the organizer's machine, and a module global it reassigned
    # would otherwise move the shot count after the cost had been printed and
    # approved. The snapshot is what makes the printed header a promise.
    team, run, n, shots, twirls = TEAM, RUN, N, SHOTS, TWIRLS
    qpu, delta, device_string = QPU, DELTA, DEVICE
    approved = (team, run, n, shots, twirls, qpu, delta, device_string)

    route = hardware_route() if device_string == "qbraid" else None
    check_environment(device_string, route)

    device_label = f"{route} via qbraid" if route else device_string
    questions = question_order(n)

    print("Qupacabrathon 2026 submission")
    print(f"  team    {team}")
    print(f"  run     {run} of 2")
    print(f"  game    odd cycle C_{n}, {n_circuits(n)} question circuits")
    print(f"  shots   {shots} per circuit, {n_circuits(n) * shots} in total")
    if twirls > 1:
        print(
            f"  twirls  {twirls} Pauli-twirled variants per question, "
            f"{shots // twirls} shots each, {twirls} task fees per question"
        )
    task_rate, shot_rate = rates_for_qpu(qpu)
    print(
        f"  qpu     {qpu}, at ${task_rate:.2f}/task + ${shot_rate:.5f}/shot, "
        f"declared deficit delta = {delta:.4f}"
    )
    print(f"  device  {device_label}")
    print(
        f"  cost    ${cost(n, shots, twirls, qpu):.2f}, {n_circuits(n)} circuits at "
        f"${cost(n, shots, twirls, qpu) / n_circuits(n):.4f} each, against the "
        f"${CAP_PER_TEAM:.2f} per-team cap across both runs"
    )
    print()

    circuits = build_circuits(n, strategy_angle(n))
    if len(circuits) != n_circuits(n):
        raise ValueError(
            f"build_circuits returned {len(circuits)} circuits, expected "
            f"{n_circuits(n)}: one per question of question_order({n})"
        )
    if (TEAM, RUN, N, SHOTS, TWIRLS, QPU, DELTA, DEVICE) != approved:
        raise RuntimeError(
            "the parameter block changed while build_circuits ran. The run that "
            "executes has to be the run that was priced and approved, so nothing "
            "was submitted. Set your parameters at the top of the file and leave "
            "them alone."
        )

    print("Sweep")
    counts = run_sweep(questions, circuits, device_string, shots, route, twirls)
    print()

    rates = [
        question_win_rate(x, y, counts[question_key(x, y)], shots) for x, y in questions
    ]
    win_rate = sum(rates) / len(rates)
    half_width = confidence_half_width(rates, shots)
    p = p_value(rates, shots, omega_c(n))
    p_nl = certified_p_nl(rates, shots, n)

    print("Result")
    print(f"  classical bound  omega_c = {omega_c(n):.6f}")
    print(f"  quantum bound    omega_q = {omega_q(n):.6f}")
    print(f"  measured         omega   = {win_rate:.6f} +/- {half_width:.6f}")
    print(f"  p-value                    {p:.3e} against the gate {P_3SIGMA:.3e}")
    if p <= P_3SIGMA:
        print(f"  CERTIFIED at 3 sigma: C_{n} beats the classical bound.")
        print(
            f"  certified nonlocal content p_NL = {p_nl:.4f} "
            f"(CHSH ceiling {CHSH_CEILING:.5f})"
        )
    else:
        print("  NOT certified at 3 sigma. This run certifies no n.")
    print()

    # In `results/`, a sibling of `submissions/` rather than of this script, in a
    # folder named for the team and the run. Run 2 cannot overwrite the evidence
    # behind run 1, the folder drops straight into the results database, and an
    # organizer moving fast at 5:00 PM never has `<team>_run1.py` sitting next to
    # `<team>_run1/`. A folder that already exists is a refusal rather than an
    # overwrite: a re-executed run is exactly the case the organizer reserve pays
    # for, and the first attempt's counts are the evidence of it.
    out_dir = Path(__file__).resolve().parents[1] / "results" / f"{team}_run{run}"
    if out_dir.exists():
        raise RuntimeError(
            f"results/{out_dir.name}/ already exists. Move it aside rather than "
            "losing what the device returned the first time."
        )
    out_dir.mkdir(parents=True)
    counts_digest = write_json(out_dir / "counts.json", counts_document(counts))
    benchmark_digest = write_json(
        out_dir / "benchmark.json",
        benchmark_document(
            win_rate, half_width, p, p_nl, device_label, team, run, n, shots, twirls
        ),
    )

    print("Wrote")
    print(f"  results/{out_dir.name}/counts.json     sha256 {counts_digest}")
    print(f"  results/{out_dir.name}/benchmark.json  sha256 {benchmark_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
