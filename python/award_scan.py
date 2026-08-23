from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


# Find the challenge repository's scripts/game_numbers.py.
ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (
        parent
        / "game_numbers.py"
    ).is_file()
)

sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

from game_numbers import (  # noqa: E402
    MINIMUM_SHOTS,
    P_3SIGMA,
    certification_power,
    cost,
    n_circuits,
    omega_c,
    omega_q,
    p_value,
    rates_for_qpu,
)


def max_affordable_shots(
    n: int,
    budget: float,
    qpu: str,
) -> int:
    """
    Maximum shots per question while staying under budget.

    TWIRLS = 1.
    """

    circuits = n_circuits(n)

    task_rate, shot_rate = (
        rates_for_qpu(qpu)
    )

    money_per_question = (
        budget / circuits
    )

    shot_money = (
        money_per_question
        - task_rate
    )

    if shot_money <= 0.0:
        return 0

    shots = math.floor(
        shot_money
        / shot_rate
    )

    # Floating-point guard.
    while (
        shots > 0
        and cost(
            n,
            shots,
            1,
            qpu,
        ) > budget
    ):
        shots -= 1

    return shots


def evaluate(
    n: int,
    shots: int,
    delta: float,
    qpu: str,
) -> dict[str, float]:

    wc = omega_c(n)
    wq = omega_q(n)

    expected = (
        wq - delta
    )

    if expected <= wc:
        return {
            "possible": 0.0,
            "power": 0.0,
            "expected_p": 1.0,
            "expected": expected,
            "cost": cost(
                n,
                shots,
                1,
                qpu,
            ),
        }

    rates = [
        expected
    ] * n_circuits(n)

    expected_p = p_value(
        rates,
        shots,
        wc,
    )

    power = certification_power(
        n,
        shots,
        delta,
    )

    return {
        "possible": 1.0,
        "power": power,
        "expected_p":
            expected_p,
        "expected":
            expected,
        "cost": cost(
            n,
            shots,
            1,
            qpu,
        ),
    }


def parse_deltas(
    text: str,
) -> list[float]:
    return [
        float(x.strip())
        for x in text.split(",")
        if x.strip()
    ]


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--budget",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--qpu",
        default="emerald",
    )

    parser.add_argument(
        "--min-n",
        type=int,
        default=13,
    )

    parser.add_argument(
        "--max-n",
        type=int,
        default=29,
    )

    parser.add_argument(
        "--deltas",
        default=(
            "0.005,"
            "0.0075,"
            "0.010,"
            "0.0125,"
            "0.015,"
            "0.0175,"
            "0.020,"
            "0.025,"
            "0.030"
        ),
    )

    parser.add_argument(
        "--power-floor",
        type=float,
        default=0.50,
    )

    args = parser.parse_args()

    deltas = parse_deltas(
        args.deltas
    )

    print()
    print(
        "ONE-RUN LARGEST-N AWARD SCAN"
    )
    print(
        "============================"
    )

    print()
    print(
        f"QPU:        {args.qpu}"
    )

    print(
        f"budget:     ${args.budget:.2f}"
    )

    print(
        f"power floor {args.power_floor:.2f}"
    )

    print()

    # --------------------------------------------------------
    # Full table
    # --------------------------------------------------------

    rows_by_delta: dict[
        float,
        list[
            tuple[
                int,
                int,
                dict[str, float],
            ]
        ],
    ] = {}

    for delta in deltas:

        print()
        print(
            f"DELTA = {delta:.4f}"
        )
        print(
            "-" * 72
        )

        print(
            " n   shots    cost"
            "      omega"
            "      margin"
            "      P(cert)"
            "      exp-p"
        )

        rows = []

        for n in range(
            args.min_n,
            args.max_n + 1,
            2,
        ):

            shots = (
                max_affordable_shots(
                    n,
                    args.budget,
                    args.qpu,
                )
            )

            if (
                shots
                < MINIMUM_SHOTS
            ):
                continue

            result = evaluate(
                n,
                shots,
                delta,
                args.qpu,
            )

            margin = (
                result["expected"]
                - omega_c(n)
            )

            rows.append(
                (
                    n,
                    shots,
                    result,
                )
            )

            print(
                f"{n:2d}"
                f"  {shots:5d}"
                f"   ${result['cost']:5.2f}"
                f"   {result['expected']:.6f}"
                f"   {margin:+.6f}"
                f"     {result['power']:.3f}"
                f"    {result['expected_p']:.2e}"
            )

        rows_by_delta[
            delta
        ] = rows

    # --------------------------------------------------------
    # Award frontier
    # --------------------------------------------------------

    print()
    print()
    print(
        "AWARD FRONTIER"
    )
    print(
        "=============="
    )

    print()
    print(
        "Largest odd n with certification "
        f"power >= {args.power_floor:.2f}"
    )

    print()

    print(
        " delta       n    shots"
        "    P(cert)     cost"
    )

    selected: dict[
        float,
        tuple[
            int,
            int,
            dict[str, float],
        ]
        | None,
    ] = {}

    for delta in deltas:

        rows = rows_by_delta[
            delta
        ]

        eligible = [
            row
            for row in rows
            if (
                row[2]["power"]
                >= args.power_floor
                and
                row[2]["expected_p"]
                <= P_3SIGMA
            )
        ]

        if not eligible:

            selected[
                delta
            ] = None

            print(
                f"{delta:7.4f}"
                "     --       --"
                "        --        --"
            )

            continue

        winner = max(
            eligible,
            key=lambda row:
                row[0],
        )

        selected[
            delta
        ] = winner

        n, shots, result = (
            winner
        )

        print(
            f"{delta:7.4f}"
            f"     {n:2d}"
            f"    {shots:5d}"
            f"       {result['power']:.3f}"
            f"      ${result['cost']:.2f}"
        )

    # --------------------------------------------------------
    # Sensitivity around each possible n
    # --------------------------------------------------------

    print()
    print()
    print(
        "INTERPRETATION"
    )
    print(
        "=============="
    )

    print()
    print(
        "The correct row is determined by "
        "the device's actual unknown DELTA."
    )

    print(
        "This script does not pretend that "
        "DELTA is measured."
    )

    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
