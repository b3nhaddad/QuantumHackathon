from __future__ import annotations

import argparse
import json
from pathlib import Path


BUILD_CIRCUITS = r'''
def build_circuits(
    n: int,
    theta: float,
) -> list[Callable[[], None]]:

    # Optimal odd-cycle strategy.
    #
    # theta = pi / (4n)
    #
    # Alice:
    #
    #     A(x) = x (pi - pi/n)
    #
    # Bob:
    #
    #     B(y) = y (pi - pi/n) + pi/(2n)

    step = math.pi - 4.0 * theta
    bob_offset = 2.0 * theta

    def gates_for(
        x: int,
        y: int,
    ) -> Callable[[], None]:

        angle_a = (
            x * step
        )

        angle_b = (
            y * step
            + bob_offset
        )

        def circuit() -> None:

            qml.Hadamard(
                wires=0
            )

            qml.CNOT(
                wires=[0, 1]
            )

            qml.RY(
                angle_a,
                wires=0,
            )

            qml.RY(
                angle_b,
                wires=1,
            )

        return circuit

    return [
        gates_for(x, y)
        for x, y
        in question_order(n)
    ]
'''.lstrip()


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "plan",
        type=Path,
    )

    parser.add_argument(
        "--team",
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "generated_emerald/"
            "build_circuits.py"
        ),
    )

    args = parser.parse_args()

    plan = json.loads(
        args.plan.read_text(
            encoding="utf-8",
        )
    )

    print()
    print("SUBMISSION PARAMETERS")
    print("=====================")
    print()

    print(
        f'TEAM = "{args.team}"'
    )

    # Required by the official event template.
    print(
        "RUN = 1"
    )

    print(
        f"N = {plan['n']}"
    )

    print(
        f"SHOTS = "
        f"{plan['shots']}"
    )

    print(
        "TWIRLS = 1"
    )

    print(
        'QPU = "emerald"'
    )

    print(
        f"DELTA = "
        f"{plan['delta']:.6f}"
    )

    print(
        'DEVICE = "default.qubit"'
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        BUILD_CIRCUITS,
        encoding="utf-8",
    )

    print()
    print(
        f"wrote {args.output}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
