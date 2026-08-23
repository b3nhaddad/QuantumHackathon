#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Import the hackathon's canonical numbers.
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(SCRIPT_DIR),
)

from game_numbers import (  # noqa: E402
    CAP_PER_TEAM,
    P_3SIGMA,
    PUBLISHED_READOUT,
    certification_power,
    cost,
    delta_from_readout_asym,
    n_circuits,
    omega_c,
    omega_q,
    p_value,
    quantum_advantage,
    rates_for_qpu,
)


# ---------------------------------------------------------------------------
# Range parsing
# ---------------------------------------------------------------------------


def parse_int_range(text: str) -> list[int]:
    """
    Accepted forms:

        21
        13,15,17,19,21
        13-29
        13-29:2
        10-200:10
    """

    values: list[int] = []

    for piece in text.split(","):
        piece = piece.strip()

        if not piece:
            continue

        if "-" not in piece:
            values.append(
                int(piece)
            )
            continue

        range_part, *step_part = piece.split(":")

        start_text, stop_text = range_part.split(
            "-",
            maxsplit=1,
        )

        start = int(start_text)
        stop = int(stop_text)

        step = (
            int(step_part[0])
            if step_part
            else 1
        )

        if step <= 0:
            raise ValueError(
                "range step must be positive"
            )

        values.extend(
            range(
                start,
                stop + 1,
                step,
            )
        )

    return sorted(
        set(values)
    )


# ---------------------------------------------------------------------------
# Device deficit assumptions
# ---------------------------------------------------------------------------


def predicted_delta(
    qpu: str,
    n: int,
) -> float:
    """
    Predict delta from the published/probed readout numbers in game_numbers.py.

    This is a model assumption, not a hardware measurement.
    """

    e0, e1, _source = (
        PUBLISHED_READOUT[qpu]
    )

    return delta_from_readout_asym(
        n,
        e0,
        e1,
    )


def choose_delta(
    *,
    qpu: str,
    n: int,
    mode: str,
    manual_delta: float | None,
) -> float:

    if mode == "published":
        return predicted_delta(
            qpu,
            n,
        )

    if mode == "manual":
        if manual_delta is None:
            raise ValueError(
                "--delta is required when "
                "--delta-mode manual"
            )

        return manual_delta

    raise ValueError(
        f"unknown delta mode {mode!r}"
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def expected_p_value(
    n: int,
    shots: int,
    expected_win_rate: float,
) -> float:
    """
    Exact event p-value evaluated at the expected outcome.

    This is not certification power.

    Power answers:
        What fraction of repeated experiments certify?

    expected_p answers:
        Does the expected outcome itself clear the 3σ gate?
    """

    if expected_win_rate <= 0.0:
        return 1.0

    if expected_win_rate >= 1.0:
        expected_win_rate = 1.0

    rates = [
        expected_win_rate
    ] * n_circuits(n)

    return p_value(
        rates,
        shots,
        omega_c(n),
    )


# ---------------------------------------------------------------------------
# One grid row
# ---------------------------------------------------------------------------


def calculate_row(
    *,
    qpu: str,
    n: int,
    shots: int,
    delta_mode: str,
    manual_delta: float | None,
) -> dict[str, object]:

    task_rate, shot_rate = (
        rates_for_qpu(qpu)
    )

    classical = omega_c(n)
    quantum = omega_q(n)

    delta = choose_delta(
        qpu=qpu,
        n=n,
        mode=delta_mode,
        manual_delta=manual_delta,
    )

    expected_win = (
        quantum
        - delta
    )

    margin = (
        expected_win
        - classical
    )

    physics_possible = (
        margin > 0.0
    )

    price = cost(
        n,
        shots,
        1,
        qpu,
    )

    within_budget = (
        price <= CAP_PER_TEAM
    )

    if physics_possible:
        power = certification_power(
            n,
            shots,
            delta,
        )

        expected_p = expected_p_value(
            n,
            shots,
            expected_win,
        )
    else:
        power = 0.0
        expected_p = 1.0

    e0, e1, source = (
        PUBLISHED_READOUT[qpu]
    )

    return {
        "qpu": qpu,

        "n": n,

        "shots_per_circuit":
            shots,

        "circuits":
            n_circuits(n),

        "total_shots":
            n_circuits(n) * shots,

        "task_rate_dollars":
            task_rate,

        "shot_rate_dollars":
            shot_rate,

        "cost_dollars":
            price,

        "within_20_dollar_cap":
            within_budget,

        "classical_bound":
            classical,

        "quantum_bound":
            quantum,

        "ideal_quantum_advantage":
            quantum_advantage(n),

        "readout_e0":
            e0,

        "readout_e1":
            e1,

        "readout_source":
            source,

        "delta_mode":
            delta_mode,

        "predicted_delta":
            delta,

        "expected_win_rate":
            expected_win,

        "margin_above_classical":
            margin,

        "physics_possible":
            physics_possible,

        "certification_power":
            power,

        "certification_power_percent":
            100.0 * power,

        "expected_p_value":
            expected_p,

        "passes_3sigma_at_expectation":
            expected_p <= P_3SIGMA,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Compare Emerald and Garnet across "
            "odd-cycle sizes and shot counts."
        )
    )

    parser.add_argument(
        "--n",
        default="3-29:2",
        help=(
            "n values. Examples: "
            "'3-29:2', '19,21,23', '21'."
        ),
    )

    parser.add_argument(
        "--shots",
        default="10-300:10",
        help=(
            "Shots per circuit. Examples: "
            "'10-300:10', '84,110,141'."
        ),
    )

    parser.add_argument(
        "--qpus",
        default="emerald,garnet",
        help=(
            "Comma-separated QPUs. "
            "Default: emerald,garnet."
        ),
    )

    parser.add_argument(
        "--delta-mode",
        choices=(
            "published",
            "manual",
        ),
        default="published",
        help=(
            "published = derive delta from each "
            "device's readout figures; "
            "manual = use the same --delta "
            "for both devices."
        ),
    )

    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help=(
            "Manual device deficit when "
            "--delta-mode manual."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "qpu_comparison.csv"
        ),
    )

    args = parser.parse_args()

    ns = parse_int_range(
        args.n
    )

    shots_values = parse_int_range(
        args.shots
    )

    qpus = [
        value.strip()
        for value in args.qpus.split(",")
        if value.strip()
    ]

    for n in ns:
        if (
            n < 3
            or n % 2 == 0
        ):
            raise ValueError(
                f"n={n} is not an odd cycle >= 3"
            )

    for shots in shots_values:
        if shots <= 0:
            raise ValueError(
                "shots must be positive"
            )

    for qpu in qpus:
        if qpu not in PUBLISHED_READOUT:
            raise ValueError(
                f"unknown QPU {qpu!r}; "
                f"available: "
                f"{sorted(PUBLISHED_READOUT)}"
            )

    total = (
        len(qpus)
        * len(ns)
        * len(shots_values)
    )

    rows: list[
        dict[str, object]
    ] = []

    progress = tqdm(
        total=total,
        desc="Comparing QPUs",
        unit="plan",
        dynamic_ncols=True,
    )

    for qpu in qpus:
        for n in ns:
            for shots in shots_values:

                rows.append(
                    calculate_row(
                        qpu=qpu,
                        n=n,
                        shots=shots,
                        delta_mode=
                            args.delta_mode,
                        manual_delta=
                            args.delta,
                    )
                )

                progress.update(1)

    progress.close()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise RuntimeError(
            "No rows generated."
        )

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print()
    print(
        f"Wrote {len(rows):,} rows "
        f"to {args.output}"
    )

    print()

    # Short summary at the end.
    for qpu in qpus:

        qpu_rows = [
            row
            for row in rows
            if row["qpu"] == qpu
        ]

        affordable = [
            row
            for row in qpu_rows
            if row[
                "within_20_dollar_cap"
            ]
        ]

        certifying = [
            row
            for row in affordable
            if row[
                "passes_3sigma_at_expectation"
            ]
        ]

        print(
            f"{qpu}:"
        )

        if not certifying:
            print(
                "  no tested plan both fits "
                "the budget and clears 3σ "
                "at its expected outcome"
            )
            continue

        highest_n = max(
            int(row["n"])
            for row in certifying
        )

        best_at_n = max(
            (
                row
                for row in certifying
                if int(row["n"])
                == highest_n
            ),
            key=lambda row:
                float(
                    row[
                        "certification_power"
                    ]
                ),
        )

        print(
            f"  highest tested n "
            f"clearing expectation: "
            f"C_{highest_n}"
        )

        print(
            f"  shots: "
            f"{best_at_n['shots_per_circuit']}"
        )

        print(
            f"  cost: "
            f"${float(best_at_n['cost_dollars']):.2f}"
        )

        print(
            f"  P(certify): "
            f"{100.0 * float(best_at_n['certification_power']):.1f}%"
        )

        print(
            f"  delta: "
            f"{float(best_at_n['predicted_delta']):.5f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
