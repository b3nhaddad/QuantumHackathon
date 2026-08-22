from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DeviceParams:
    """Low-dimensional model of the IQM execution path after RY folding.

    All coherent parameters are small-error parameters around the exact folded
    circuit used by the challenge template.

    relative_offset:
        Relative additive angle bias between Alice and Bob, in radians.
        +d means Alice receives +d/2 and Bob -d/2.
    scale_a / scale_b:
        Fractional scale error on the *folded RY remainder* for each player.
    yerr_a / yerr_b:
        Coherent overrotation, in radians, on the inserted Pauli-Y half turn.
        It only contributes when the challenge's RY-fold transform inserts Y.
    visibility:
        Werner-state visibility. 1 is an ideal Bell state, 0 is maximally mixed.
    e0 / e1:
        Shared per-wire asymmetric readout error: 0->1 and 1->0 respectively.
    """

    relative_offset: float = 0.0
    scale_a: float = 0.0
    scale_b: float = 0.0
    yerr_a: float = 0.0
    yerr_b: float = 0.0
    visibility: float = 1.0
    e0: float = 0.0
    e1: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                self.relative_offset,
                self.scale_a,
                self.scale_b,
                self.yerr_a,
                self.yerr_b,
                self.visibility,
                self.e0,
                self.e1,
            ],
            dtype=float,
        )

    @staticmethod
    def from_array(values: Iterable[float]) -> "DeviceParams":
        v = list(values)
        if len(v) != 8:
            raise ValueError(f"expected 8 device parameters, got {len(v)}")
        return DeviceParams(*map(float, v))


PARAMETER_NAMES = (
    "relative_offset",
    "scale_a",
    "scale_b",
    "yerr_a",
    "yerr_b",
    "visibility",
    "e0",
    "e1",
)


@dataclass(frozen=True)
class ProbeDesign:
    """A legal, low-dimensional perturbation of the odd-cycle angle strategy.

    The perturbations preserve locality: Alice's requested angle depends only on
    x and Bob's only on y.
    """

    relative_offset: float = 0.0
    ramp_a: float = 0.0
    ramp_b: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.relative_offset, self.ramp_a, self.ramp_b], dtype=float)


QPU_READOUT = {
    # Values published/recorded by the challenge repository. These are priors,
    # not measured device deficits.
    "emerald": (0.0013, 0.0280),
    "garnet": (0.0300, 0.0300),
}

QPU_RATES = {
    "emerald": (0.30, 0.00160),
    "garnet": (0.30, 0.00145),
}


# Prior widths are OUR modeling choices for run-1 design, not event facts.
# Information-gain scoring standardizes by these widths, so they should encode
# what ranges are worth learning rather than claim those errors are present.
DEFAULT_PRIOR_SIGMA = np.array(
    [
        0.08,   # relative angle offset [rad]
        0.03,   # Alice RY scale
        0.03,   # Bob RY scale
        0.05,   # Alice Y overrotation [rad]
        0.05,   # Bob Y overrotation [rad]
        0.015,  # visibility
        0.010,  # e0
        0.015,  # e1
    ],
    dtype=float,
)


def nominal_params(qpu: str) -> DeviceParams:
    if qpu not in QPU_READOUT:
        raise ValueError(f"unknown QPU {qpu!r}; choose one of {sorted(QPU_READOUT)}")
    e0, e1 = QPU_READOUT[qpu]
    # Keep visibility close to one. It is intentionally uncertain in the prior;
    # the event does not publish a device deficit.
    return DeviceParams(visibility=0.995, e0=e0, e1=e1)


def omega_c(n: int) -> float:
    return 1.0 - 1.0 / (2.0 * n)


def omega_q(n: int) -> float:
    return math.cos(math.pi / (4.0 * n)) ** 2


def strategy_theta(n: int) -> float:
    return math.pi / (4.0 * n)


def question_order(n: int) -> list[tuple[int, int]]:
    return [(i, i) for i in range(n)] + [(i, (i + 1) % n) for i in range(n)]


def base_angle_functions(n: int) -> tuple[np.ndarray, np.ndarray]:
    """The optimal noiseless odd-cycle strategy used in the submission.

    A_x = x (pi - pi/n)
    B_y = y (pi - pi/n) + pi/(2n)
    """
    theta = strategy_theta(n)
    step = math.pi - 4.0 * theta
    a = np.array([x * step for x in range(n)], dtype=float)
    b = np.array([y * step + 2.0 * theta for y in range(n)], dtype=float)
    return a, b


def requested_angles(n: int, probe: ProbeDesign) -> tuple[np.ndarray, np.ndarray]:
    a, b = base_angle_functions(n)
    if n == 1:
        u = np.zeros(1)
    else:
        u = np.linspace(-1.0, 1.0, n)

    # A relative offset is split symmetrically only to avoid choosing a preferred
    # player. Only the difference matters to the ideal Bell-pair probabilities.
    a = a + 0.5 * probe.relative_offset + probe.ramp_a * u
    b = b - 0.5 * probe.relative_offset + probe.ramp_b * u
    return a, b


def fold_ry(angle: float) -> tuple[float, int]:
    """Mirror challenge/submission-template.py::fold_ry_angles.

    Returns (remainder, y_parity), where the exact ideal operation is equivalent
    up to global phase to RY(remainder) followed by Y when y_parity == 1.
    """
    remainder = math.remainder(float(angle), math.pi)
    half_turns = round((float(angle) - remainder) / math.pi)
    return remainder, half_turns % 2


def effective_angle(requested: float, scale: float, yerr: float, offset: float) -> float:
    """Effective Y-axis rotation under the small coherent hardware model."""
    remainder, y_parity = fold_ry(requested)
    return (1.0 + scale) * remainder + y_parity * (math.pi + yerr) + offset


def _validate_params(p: DeviceParams) -> None:
    if not 0.0 < p.visibility <= 1.0:
        raise ValueError(f"visibility must be in (0, 1], got {p.visibility}")
    for name, value in (("e0", p.e0), ("e1", p.e1)):
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be in [0, 1), got {value}")


def outcome_probabilities(
    a_requested: float,
    b_requested: float,
    params: DeviceParams,
) -> np.ndarray:
    """Return [P00, P01, P10, P11] for one question.

    The state model is a Werner mixture around |Phi+>. The challenge's folded
    RY execution is modeled explicitly, then the same asymmetric confusion
    matrix is applied independently to both measured wires.
    """
    _validate_params(params)

    a_eff = effective_angle(
        a_requested,
        params.scale_a,
        params.yerr_a,
        +0.5 * params.relative_offset,
    )
    b_eff = effective_angle(
        b_requested,
        params.scale_b,
        params.yerr_b,
        -0.5 * params.relative_offset,
    )

    delta = a_eff - b_eff
    agree = math.cos(delta / 2.0) ** 2

    v = params.visibility
    p00 = 0.5 * v * agree + 0.25 * (1.0 - v)
    p11 = p00
    p01 = 0.5 * v * (1.0 - agree) + 0.25 * (1.0 - v)
    p10 = p01
    true = np.array([[p00, p01], [p10, p11]], dtype=float)

    # confusion[observed, true]
    confusion = np.array(
        [
            [1.0 - params.e0, params.e1],
            [params.e0, 1.0 - params.e1],
        ],
        dtype=float,
    )
    observed = confusion @ true @ confusion.T
    flat = observed.reshape(4)
    flat /= flat.sum()
    return flat


def sweep_probabilities(
    n: int,
    probe: ProbeDesign,
    params: DeviceParams,
) -> np.ndarray:
    a, b = requested_angles(n, probe)
    return np.array(
        [outcome_probabilities(a[x], b[y], params) for x, y in question_order(n)],
        dtype=float,
    )


def question_win_rates(n: int, probabilities: np.ndarray) -> np.ndarray:
    rates: list[float] = []
    for (x, y), row in zip(question_order(n), probabilities, strict=True):
        if x == y:
            rates.append(float(row[0] + row[3]))
        else:
            rates.append(float(row[1] + row[2]))
    return np.array(rates, dtype=float)


def mean_win_rate(n: int, probe: ProbeDesign, params: DeviceParams) -> float:
    return float(np.mean(question_win_rates(n, sweep_probabilities(n, probe, params))))


def cost(n: int, shots: int, qpu: str, twirls: int = 1) -> float:
    task, shot = QPU_RATES[qpu]
    return 2.0 * n * (task * twirls + shot * shots)


def shots_for_budget(n: int, budget: float, qpu: str, twirls: int = 1) -> int:
    task, shot = QPU_RATES[qpu]
    per_question = budget / (2.0 * n)
    return math.floor((per_question - task * twirls) / shot)


def readout_delta_formula(n: int, e0: float, e1: float) -> float:
    """Repository's asymmetric-readout mean-deficit formula."""
    ebar = 0.5 * (e0 + e1)
    return 2.0 * ebar * (1.0 - ebar) * (2.0 * omega_q(n) - 1.0)

