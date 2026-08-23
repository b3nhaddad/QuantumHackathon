from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from scipy.stats import binom

from emerald_noise_model import (
    P_3SIGMA,
    build_emerald_latent_prior,
    choose_backend,
    emerald_cost,
    omega_c,
    omega_q,
    question_probabilities_from_tables,
    question_win_rates,
    shots_for_budget,
    textbook_angles,
)

MINIMUM_SHOTS = 10


def sample_latent_gaussian(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    count: int,
    *,
    seed: int,
) -> torch.Tensor:
    """
    Draw latent Emerald parameter samples from the Gaussian approximation
    produced by build_emerald_latent_prior().

    This is used only for local Monte Carlo planning.
    """

    generator = torch.Generator(
        device=mean.device
    )

    generator.manual_seed(
        seed
    )

    eps = torch.randn(
        (
            count,
            mean.numel(),
        ),
        generator=generator,
        device=mean.device,
        dtype=mean.dtype,
    )

    eye = torch.eye(
        mean.numel(),
        device=mean.device,
        dtype=mean.dtype,
    )

    # Small numerical jitter keeps the covariance positive definite
    # in float32.
    covariance = (
        covariance
        + 1e-10 * eye
    )

    chol = torch.linalg.cholesky(
        covariance
    )

    return (
        mean.unsqueeze(0)
        + eps @ chol.T
    )


# --------------------------------------------------------------------------
# Event certification gate
# --------------------------------------------------------------------------


def critical_wins(
    n: int,
    shots: int,
) -> int:
    """
    Smallest pooled win count that clears the event's one-sided
    exact binomial 3-sigma test.
    """

    trials = 2 * n * shots
    null_rate = omega_c(n)

    lo = 0
    hi = trials

    while lo < hi:
        mid = (lo + hi) // 2

        p = binom.sf(
            mid - 1,
            trials,
            null_rate,
        )

        if p <= P_3SIGMA:
            hi = mid
        else:
            lo = mid + 1

    return lo


# --------------------------------------------------------------------------
# Cost
# --------------------------------------------------------------------------


def maximum_affordable_shots(
    n: int,
    budget: float,
) -> int:
    """
    Spend essentially the whole budget at this n.

    TWIRLS = 1.
    """

    shots = shots_for_budget(
        n,
        budget,
        twirls=1,
    )

    while (
        shots >= MINIMUM_SHOTS
        and emerald_cost(
            n,
            shots,
            1,
        ) > budget
    ):
        shots -= 1

    return shots


# --------------------------------------------------------------------------
# Certification probability
# --------------------------------------------------------------------------


def conditional_certification_probability(
    rates: torch.Tensor,
    n: int,
    shots: int,
) -> torch.Tensor:
    """
    Approximate P(certify | one particular Emerald latent state).

    `rates` is [D, 2n].

    Every question contributes Binomial(shots, p_q).

    The certification threshold itself is exact.
    We use a continuity-corrected normal approximation only for the
    alternative-distribution power calculation.
    """

    threshold = critical_wins(
        n,
        shots,
    )

    mean = (
        shots
        * rates.sum(
            dim=-1
        )
    )

    variance = (
        shots
        * (
            rates
            * (1.0 - rates)
        ).sum(
            dim=-1
        )
    )

    sigma = torch.sqrt(
        variance.clamp_min(
            1e-12
        )
    )

    z = (
        threshold
        - 0.5
        - mean
    ) / sigma

    return (
        0.5
        * torch.erfc(
            z
            / math.sqrt(2.0)
        )
    )


def evaluate_n(
    *,
    n: int,
    shots: int,
    latents: torch.Tensor,
    latent_chunk: int,
) -> dict[str, float]:

    powers: list[torch.Tensor] = []
    mean_rates: list[torch.Tensor] = []

    alice, bob = textbook_angles(
        n,
        device=latents.device,
        dtype=latents.dtype,
    )

    for start in range(
        0,
        latents.shape[0],
        latent_chunk,
    ):
        latent = latents[
            start:
            start + latent_chunk
        ]

        probs = (
            question_probabilities_from_tables(
                n,
                alice,
                bob,
                latent,
            )
        )

        rates = question_win_rates(
            n,
            probs,
        )

        # Current Emerald model returns:
        #
        #     [latent, design, question]
        #
        # for a single design.
        while (
            rates.ndim > 2
            and rates.shape[1] == 1
        ):
            rates = rates.squeeze(1)

        power = (
            conditional_certification_probability(
                rates,
                n,
                shots,
            )
        )

        powers.append(
            power.detach()
        )

        mean_rates.append(
            rates.mean(
                dim=-1
            ).detach()
        )

    power = torch.cat(
        powers
    )

    mean_rate = torch.cat(
        mean_rates
    )

    deficit = (
        omega_q(n)
        - mean_rate
    )

    return {
        "predictive_cert_probability":
            float(
                power.mean().item()
            ),

        "power_q10":
            float(
                torch.quantile(
                    power,
                    0.10,
                ).item()
            ),

        "power_median":
            float(
                torch.quantile(
                    power,
                    0.50,
                ).item()
            ),

        "power_q90":
            float(
                torch.quantile(
                    power,
                    0.90,
                ).item()
            ),

        "mean_win_rate":
            float(
                mean_rate.mean().item()
            ),

        "median_win_rate":
            float(
                torch.quantile(
                    mean_rate,
                    0.50,
                ).item()
            ),

        "mean_delta":
            float(
                deficit.mean().item()
            ),

        "median_delta":
            float(
                torch.quantile(
                    deficit,
                    0.50,
                ).item()
            ),

        "delta_q75":
            float(
                torch.quantile(
                    deficit,
                    0.75,
                ).item()
            ),
    }


# --------------------------------------------------------------------------
# Main search
# --------------------------------------------------------------------------


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--budget",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--min-n",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-n",
        type=int,
        default=31,
    )

    parser.add_argument(
        "--draws",
        type=int,
        default=262_144,
    )

    parser.add_argument(
        "--physical-draws",
        type=int,
        default=262_144,
    )

    parser.add_argument(
        "--latent-chunk",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--dtype",
        default="float32",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260822,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "generated_emerald/"
            "award_frontier.json"
        ),
    )

    args = parser.parse_args()

    backend = choose_backend(
        args.device,
        args.dtype,
    )

    print(
        f"backend: {backend.device}"
    )

    # ----------------------------------------------------------
    # Emerald prior
    # ----------------------------------------------------------

    prior = (
        build_emerald_latent_prior(
            backend=backend,
            physical_draws=
                args.physical_draws,
            seed=args.seed,
        )
    )

    latents = sample_latent_gaussian(
        prior.mean,
        prior.covariance,
        args.draws,
        seed=args.seed + 1,
    )

    # Gaussian approximation can occasionally generate
    # nonsensical readout probabilities.
    latents[:, 13:17] = (
        latents[:, 13:17].clamp(
            1e-7,
            0.15,
        )
    )

    print()
    print(
        "Largest-certified-n one-run search"
    )
    print(
        "=================================="
    )
    print()

    table: list[
        dict[str, object]
    ] = []

    for n in range(
        args.min_n,
        args.max_n + 1,
        2,
    ):

        shots = (
            maximum_affordable_shots(
                n,
                args.budget,
            )
        )

        if shots < MINIMUM_SHOTS:
            continue

        stats = evaluate_n(
            n=n,
            shots=shots,
            latents=latents,
            latent_chunk=
                args.latent_chunk,
        )

        cost = emerald_cost(
            n,
            shots,
            1,
        )

        row = {
            "n": n,
            "shots": shots,
            "twirls": 1,
            "cost": cost,

            "classical_bound":
                omega_c(n),

            "quantum_bound":
                omega_q(n),

            **stats,
        }

        table.append(
            row
        )

        print(
            f"C_{n:<2}  "
            f"S={shots:<4}  "
            f"${cost:>5.2f}  "
            f"Pcert="
            f"{stats['predictive_cert_probability']:.3f}  "
            f"omega="
            f"{stats['mean_win_rate']:.6f}  "
            f"delta="
            f"{stats['median_delta']:.5f}"
        )

    # ----------------------------------------------------------
    # Show award frontier at several risk tolerances
    # ----------------------------------------------------------

    risk_levels = [
        0.90,
        0.75,
        0.60,
        0.50,
        0.40,
        0.30,
    ]

    frontiers: dict[
        str,
        dict[str, object] | None
    ] = {}

    print()
    print("Award frontier")
    print("==============")
    print()

    for floor in risk_levels:

        eligible = [
            row
            for row in table
            if (
                row[
                    "predictive_cert_probability"
                ]
                >= floor
            )
        ]

        if not eligible:
            chosen = None

            print(
                f"P >= {floor:.2f}: "
                "no candidate"
            )

        else:
            chosen = max(
                eligible,
                key=lambda row:
                    int(row["n"]),
            )

            print(
                f"P >= {floor:.2f}: "
                f"C_{chosen['n']}  "
                f"{chosen['shots']} shots  "
                f"P="
                f"{chosen['predictive_cert_probability']:.3f}"
            )

        frontiers[
            f"{floor:.2f}"
        ] = chosen

    document = {
        "objective":
            "largest_certified_n",

        "hardware_runs": 1,

        "budget":
            args.budget,

        "qpu":
            "emerald",

        "prior_physical_draws":
            args.physical_draws,

        "predictive_draws":
            args.draws,

        "risk_frontiers":
            frontiers,

        "candidates":
            table,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            document,
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
