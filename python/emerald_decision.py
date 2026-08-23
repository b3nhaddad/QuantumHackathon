from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "frontier",
        type=Path,
    )

    parser.add_argument(
        "--min-cert-prob",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "generated_emerald/"
            "selected_plan.json"
        ),
    )

    args = parser.parse_args()

    document = json.loads(
        args.frontier.read_text(
            encoding="utf-8",
        )
    )

    candidates = document[
        "candidates"
    ]

    eligible = [
        candidate
        for candidate in candidates
        if float(
            candidate[
                "predictive_cert_probability"
            ]
        )
        >= args.min_cert_prob
    ]

    if not eligible:
        raise RuntimeError(
            "No candidate reaches "
            f"P(certify) >= "
            f"{args.min_cert_prob:.3f}"
        )

    # Award objective:
    #
    #   largest certified n
    #
    # Therefore n is the primary key.
    chosen = max(
        eligible,
        key=lambda c: (
            int(c["n"]),
            float(
                c[
                    "predictive_cert_probability"
                ]
            ),
        ),
    )

    print()
    print("ONE-RUN AWARD PLAN")
    print("==================")
    print()

    print(
        f"N          "
        f"{chosen['n']}"
    )

    print(
        f"SHOTS      "
        f"{chosen['shots']}"
    )

    print(
        "TWIRLS     1"
    )

    print(
        "QPU        emerald"
    )

    print(
        f"cost       "
        f"${chosen['cost']:.2f}"
    )

    print(
        f"P(certify) "
        f"{chosen['predictive_cert_probability']:.3f}"
    )

    print(
        f"median Δ   "
        f"{chosen['median_delta']:.6f}"
    )

    print(
        f"75% Δ      "
        f"{chosen['delta_q75']:.6f}"
    )

    # For an aggressive one-shot entry I would use the
    # model's median predicted deficit as DELTA.
    #
    # It is an assumption, not a hardware measurement.
    selected = {
        "objective":
            "largest_certified_n",

        "hardware_runs":
            1,

        "run":
            1,

        "n":
            chosen["n"],

        "shots":
            chosen["shots"],

        "twirls":
            1,

        "qpu":
            "emerald",

        "delta":
            chosen["median_delta"],

        "predictive_cert_probability":
            chosen[
                "predictive_cert_probability"
            ],

        "cost":
            chosen["cost"],
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            selected,
            indent=2,
        )
        + "\n",
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
