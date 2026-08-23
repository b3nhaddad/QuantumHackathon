from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from emerald_noise_model import (
    LATENT_NAMES,
    build_emerald_latent_prior,
    choose_backend,
    latent_probabilities,
    omega_q,
    question_order,
    question_win_rates,
    textbook_angles,
)


# ---------------------------------------------------------------------------
# Gaussian latent sampling
# ---------------------------------------------------------------------------


def sample_latent_gaussian(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    count: int,
    *,
    seed: int,
) -> torch.Tensor:
    """
    Draw local Monte-Carlo samples from the compact Emerald latent prior.

    This is intentionally kept in the optimizer rather than the physical model.
    """

    generator = torch.Generator(
        device=mean.device,
    )

    generator.manual_seed(
        seed,
    )

    dim = mean.numel()

    eps = torch.randn(
        (count, dim),
        generator=generator,
        device=mean.device,
        dtype=mean.dtype,
    )

    eye = torch.eye(
        dim,
        device=mean.device,
        dtype=mean.dtype,
    )

    chol = None

    for jitter in (
        1e-10,
        1e-9,
        1e-8,
        1e-7,
        1e-6,
    ):
        candidate, info = torch.linalg.cholesky_ex(
            covariance + jitter * eye
        )

        if int(info.max().item()) == 0:
            chol = candidate
            break

    if chol is None:
        raise RuntimeError(
            "Could not Cholesky-factor Emerald latent covariance."
        )

    return (
        mean.unsqueeze(0)
        + eps @ chol.T
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def wrap_gauge(
    gauge: float,
) -> float:
    """
    Gauge is 2*pi periodic under the compiler-aware folding model.
    """

    return (
        (gauge + math.pi)
        % (2.0 * math.pi)
        - math.pi
    )


def clamp_latents(
    latent: torch.Tensor,
) -> torch.Tensor:
    """
    Gaussian latent draws can place readout probabilities just outside
    physical range.

    Clamp only those probability-like coordinates.
    """

    latent = latent.clone()

    readout_names = (
        "e0_a",
        "e1_a",
        "e0_b",
        "e1_b",
    )

    for name in readout_names:
        index = LATENT_NAMES.index(name)

        latent[:, index] = (
            latent[:, index].clamp(
                1e-7,
                0.15,
            )
        )

    return latent


# ---------------------------------------------------------------------------
# Angle families
# ---------------------------------------------------------------------------


def textbook_question_angles(
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Textbook +chirality, zero-gauge reference.
    """

    alice, bob = textbook_angles(
        n,
        device=device,
        dtype=dtype,
    )

    x, y, _ = question_order(
        n,
        device=device,
    )

    return (
        alice[x],
        bob[y],
    )


def candidate_question_angles(
    n: int,
    gauges: torch.Tensor,
    chiralities: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Build requested RY angles for a batch of exact-equivalent
    odd-cycle strategies.

    gauges:       [B]
    chiralities:  [B], values +/-1

    Returns:
        Alice requested angles [B, 2n]
        Bob requested angles   [B, 2n]
    """

    device = gauges.device
    dtype = gauges.dtype

    theta = (
        math.pi
        / (4.0 * n)
    )

    base_step = (
        math.pi
        - 4.0 * theta
    )

    idx = torch.arange(
        n,
        device=device,
        dtype=dtype,
    )

    chirality = (
        chiralities
        .to(dtype=dtype)
        .reshape(-1, 1)
    )

    gauge = gauges.reshape(
        -1,
        1,
    )

    alice_table = (
        gauge
        + chirality
        * idx.reshape(1, -1)
        * base_step
    )

    bob_table = (
        gauge
        + chirality
        * idx.reshape(1, -1)
        * base_step
        + chirality
        * (2.0 * theta)
    )

    x, y, _ = question_order(
        n,
        device=device,
    )

    return (
        alice_table[:, x],
        bob_table[:, y],
    )


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def baseline_mean_win_rates(
    *,
    n: int,
    latents: torch.Tensor,
    latent_chunk: int,
) -> torch.Tensor:
    """
    Mean win rate for the textbook strategy for every latent draw.

    Output:
        [D]
    """

    a, b = textbook_question_angles(
        n,
        device=latents.device,
        dtype=latents.dtype,
    )

    pieces: list[torch.Tensor] = []

    for start in range(
        0,
        latents.shape[0],
        latent_chunk,
    ):
        latent = latents[
            start:
            start + latent_chunk
        ]

        probabilities = latent_probabilities(
            a,
            b,
            latent,
        )

        rates = question_win_rates(
            n,
            probabilities,
        )

        pieces.append(
            rates.mean(
                dim=-1
            )
        )

    return torch.cat(
        pieces,
        dim=0,
    )


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------


def evaluate_candidates(
    *,
    n: int,
    gauges: list[float],
    chiralities: list[int],
    latents: torch.Tensor,
    baseline: torch.Tensor,
    candidate_batch: int,
    latent_chunk: int,
) -> list[dict[str, float | int]]:

    if len(gauges) != len(chiralities):
        raise ValueError(
            "gauges and chiralities must have the same length"
        )

    records: list[
        dict[str, float | int]
    ] = []

    device = latents.device
    dtype = latents.dtype

    for candidate_start in range(
        0,
        len(gauges),
        candidate_batch,
    ):
        candidate_end = min(
            candidate_start
            + candidate_batch,
            len(gauges),
        )

        gauge_tensor = torch.tensor(
            gauges[
                candidate_start:
                candidate_end
            ],
            device=device,
            dtype=dtype,
        )

        chirality_tensor = torch.tensor(
            chiralities[
                candidate_start:
                candidate_end
            ],
            device=device,
            dtype=torch.int64,
        )

        a, b = candidate_question_angles(
            n,
            gauge_tensor,
            chirality_tensor,
        )

        candidate_pieces: list[
            torch.Tensor
        ] = []

        uplift_pieces: list[
            torch.Tensor
        ] = []

        for latent_start in range(
            0,
            latents.shape[0],
            latent_chunk,
        ):
            latent_end = min(
                latent_start
                + latent_chunk,
                latents.shape[0],
            )

            latent = latents[
                latent_start:
                latent_end
            ]

            probabilities = latent_probabilities(
                a,
                b,
                latent,
            )

            rates = question_win_rates(
                n,
                probabilities,
            )

            mean_rate = rates.mean(
                dim=-1
            )

            base = baseline[
                latent_start:
                latent_end
            ].reshape(
                -1,
                1,
            )

            uplift = (
                mean_rate
                - base
            )

            candidate_pieces.append(
                mean_rate
            )

            uplift_pieces.append(
                uplift
            )

        candidate_rate = torch.cat(
            candidate_pieces,
            dim=0,
        )

        uplift = torch.cat(
            uplift_pieces,
            dim=0,
        )

        mean_uplift = uplift.mean(
            dim=0
        )

        median_uplift = torch.quantile(
            uplift,
            0.50,
            dim=0,
        )

        q25_uplift = torch.quantile(
            uplift,
            0.25,
            dim=0,
        )

        q10_uplift = torch.quantile(
            uplift,
            0.10,
            dim=0,
        )

        q01_uplift = torch.quantile(
            uplift,
            0.01,
            dim=0,
        )

        p_beat = (
            uplift > 1e-9
        ).to(
            dtype
        ).mean(
            dim=0
        )

        mean_win = candidate_rate.mean(
            dim=0
        )

        median_win = torch.quantile(
            candidate_rate,
            0.50,
            dim=0,
        )

        for local_index in range(
            candidate_end
            - candidate_start
        ):
            global_index = (
                candidate_start
                + local_index
            )

            records.append(
                {
                    "gauge":
                        float(
                            gauges[
                                global_index
                            ]
                        ),

                    "chirality":
                        int(
                            chiralities[
                                global_index
                            ]
                        ),

                    "mean_uplift":
                        float(
                            mean_uplift[
                                local_index
                            ].item()
                        ),

                    "median_uplift":
                        float(
                            median_uplift[
                                local_index
                            ].item()
                        ),

                    "q25_uplift":
                        float(
                            q25_uplift[
                                local_index
                            ].item()
                        ),

                    "q10_uplift":
                        float(
                            q10_uplift[
                                local_index
                            ].item()
                        ),

                    "q01_uplift":
                        float(
                            q01_uplift[
                                local_index
                            ].item()
                        ),

                    "p_beat_textbook":
                        float(
                            p_beat[
                                local_index
                            ].item()
                        ),

                    "mean_win_rate":
                        float(
                            mean_win[
                                local_index
                            ].item()
                        ),

                    "median_win_rate":
                        float(
                            median_win[
                                local_index
                            ].item()
                        ),
                }
            )

    return records


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def ranking_key(
    record: dict[str, float | int],
    metric: str,
) -> tuple[float, ...]:

    if metric == "q10":
        return (
            float(record["q10_uplift"]),
            float(record["median_uplift"]),
            float(record["mean_uplift"]),
            float(record["p_beat_textbook"]),
        )

    if metric == "q25":
        return (
            float(record["q25_uplift"]),
            float(record["median_uplift"]),
            float(record["mean_uplift"]),
        )

    if metric == "median":
        return (
            float(record["median_uplift"]),
            float(record["q10_uplift"]),
            float(record["mean_uplift"]),
        )

    if metric == "mean":
        return (
            float(record["mean_uplift"]),
            float(record["q10_uplift"]),
            float(record["median_uplift"]),
        )

    if metric == "pbeat":
        return (
            float(record["p_beat_textbook"]),
            float(record["q10_uplift"]),
            float(record["mean_uplift"]),
        )

    raise ValueError(
        f"unknown ranking metric {metric!r}"
    )


def top_records(
    records: list[
        dict[str, float | int]
    ],
    *,
    metric: str,
    count: int,
) -> list[
    dict[str, float | int]
]:
    return sorted(
        records,
        key=lambda record:
            ranking_key(
                record,
                metric,
            ),
        reverse=True,
    )[:count]


# ---------------------------------------------------------------------------
# Search grids
# ---------------------------------------------------------------------------


def coarse_candidates(
    points: int,
) -> tuple[
    list[float],
    list[int],
]:

    gauges: list[float] = []
    chiralities: list[int] = []

    # Include gauge = 0 exactly.
    spacing = (
        2.0
        * math.pi
        / points
    )

    for chirality in (
        +1,
        -1,
    ):
        for index in range(
            points
        ):
            gauge = (
                -math.pi
                + index
                * spacing
            )

            gauges.append(
                wrap_gauge(
                    gauge
                )
            )

            chiralities.append(
                chirality
            )

        gauges.append(
            0.0
        )

        chiralities.append(
            chirality
        )

    return (
        gauges,
        chiralities,
    )


def refinement_candidates(
    coarse_top: list[
        dict[str, float | int]
    ],
    *,
    coarse_points: int,
    refine_points: int,
) -> tuple[
    list[float],
    list[int],
]:

    coarse_spacing = (
        2.0
        * math.pi
        / coarse_points
    )

    unique: dict[
        tuple[int, int],
        float,
    ] = {}

    for record in coarse_top:

        center = float(
            record["gauge"]
        )

        chirality = int(
            record["chirality"]
        )

        for i in range(
            refine_points
        ):
            if refine_points == 1:
                offset = 0.0
            else:
                fraction = (
                    i
                    / (
                        refine_points
                        - 1
                    )
                )

                offset = (
                    -coarse_spacing
                    + 2.0
                    * coarse_spacing
                    * fraction
                )

            gauge = wrap_gauge(
                center
                + offset
            )

            # Quantized key only for deduplication.
            key = (
                chirality,
                round(
                    gauge
                    * 1e10
                ),
            )

            unique[key] = gauge

    gauges: list[float] = []
    chiralities: list[int] = []

    for (
        chirality,
        _,
    ), gauge in unique.items():

        gauges.append(
            gauge
        )

        chiralities.append(
            chirality
        )

    return (
        gauges,
        chiralities,
    )


# ---------------------------------------------------------------------------
# Ideal verification
# ---------------------------------------------------------------------------


def ideal_win_rates(
    n: int,
    gauge: float,
    chirality: int,
) -> list[float]:

    theta = (
        math.pi
        / (4.0 * n)
    )

    step = (
        chirality
        * (
            math.pi
            - 4.0 * theta
        )
    )

    offset = (
        chirality
        * 2.0
        * theta
    )

    def alice(
        x: int,
    ) -> float:
        return (
            gauge
            + x * step
        )

    def bob(
        y: int,
    ) -> float:
        return (
            gauge
            + y * step
            + offset
        )

    rates: list[float] = []

    # Vertex questions.
    for x in range(
        n
    ):
        difference = (
            alice(x)
            - bob(x)
        )

        agreement = (
            math.cos(
                difference / 2.0
            )
            ** 2
        )

        rates.append(
            agreement
        )

    # Edge questions.
    for x in range(
        n
    ):
        y = (
            x + 1
        ) % n

        difference = (
            alice(x)
            - bob(y)
        )

        agreement = (
            math.cos(
                difference / 2.0
            )
            ** 2
        )

        rates.append(
            1.0
            - agreement
        )

    return rates


# ---------------------------------------------------------------------------
# Circuit generation
# ---------------------------------------------------------------------------


def render_build_circuits(
    *,
    gauge: float,
    chirality: int,
    direction: str,
) -> str:

    if direction == "forward":
        preparation = """\
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
"""

    elif direction == "reverse":
        preparation = """\
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[1, 0])
"""

    else:
        raise ValueError(
            "direction must be forward or reverse"
        )

    return f'''\
def build_circuits(
    n: int,
    theta: float,
) -> list[Callable[[], None]]:

    # Exact-equivalent odd-cycle strategy selected by the local optimizer.
    #
    # gauge      = {gauge:.16f}
    # chirality  = {chirality:+d}
    # CNOT       = {direction}
    #
    # Gauge and chirality do not change the noiseless game value.
    gauge = {gauge:.16f}
    chirality = {chirality:+d}

    step = (
        chirality
        * (
            math.pi
            - 4.0 * theta
        )
    )

    bob_offset = (
        chirality
        * 2.0
        * theta
    )

    def gates_for(
        x: int,
        y: int,
    ) -> Callable[[], None]:

        angle_a = (
            gauge
            + x * step
        )

        angle_b = (
            gauge
            + y * step
            + bob_offset
        )

        def circuit() -> None:
{preparation}
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
'''


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def print_top(
    title: str,
    records: list[
        dict[str, float | int]
    ],
    *,
    metric: str,
    count: int = 10,
) -> None:

    print()
    print(title)
    print(
        "=" * len(title)
    )

    print()
    print(
        " rank"
        " chir"
        "       gauge"
        "      q10"
        "      median"
        "       mean"
        "     P(beat)"
    )

    for rank, record in enumerate(
        top_records(
            records,
            metric=metric,
            count=count,
        ),
        1,
    ):

        print(
            f"{rank:5d}"
            f" {int(record['chirality']):+4d}"
            f" {float(record['gauge']):+11.7f}"
            f" {float(record['q10_uplift']):+10.7f}"
            f" {float(record['median_uplift']):+11.7f}"
            f" {float(record['mean_uplift']):+10.7f}"
            f" {float(record['p_beat_textbook']):9.3f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n",
        type=int,
        default=21,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--dtype",
        choices=(
            "float32",
            "float64",
        ),
        default="float32",
    )

    parser.add_argument(
        "--physical-draws",
        type=int,
        default=262_144,
    )

    parser.add_argument(
        "--latent-draws",
        type=int,
        default=262_144,
    )

    parser.add_argument(
        "--coarse-draws",
        type=int,
        default=16_384,
    )

    parser.add_argument(
        "--fine-draws",
        type=int,
        default=65_536,
    )

    parser.add_argument(
        "--gauge-points",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--coarse-keep",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--refine-points",
        type=int,
        default=129,
    )

    parser.add_argument(
        "--final-keep",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--candidate-batch",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--latent-chunk",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--metric",
        choices=(
            "q10",
            "q25",
            "median",
            "mean",
            "pbeat",
        ),
        default="q10",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260822,
    )

    parser.add_argument(
        "--preferred-direction",
        choices=(
            "forward",
            "reverse",
        ),
        default="forward",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "generated_equivalent"
        ),
    )

    args = parser.parse_args()

    if (
        args.n < 3
        or args.n % 2 == 0
    ):
        raise ValueError(
            "--n must be an odd integer >= 3"
        )

    backend = choose_backend(
        args.device,
        args.dtype,
    )

    print()
    print(
        f"backend = {backend.device}"
    )

    print(
        f"N       = {args.n}"
    )

    print(
        f"metric  = {args.metric}"
    )

    # --------------------------------------------------------
    # Build compact Emerald prior
    # --------------------------------------------------------

    print()
    print(
        "Building Emerald latent prior..."
    )

    prior = build_emerald_latent_prior(
        backend=backend,
        physical_draws=
            args.physical_draws,
        seed=args.seed,
    )

    print(
        "Sampling latent device draws..."
    )

    latents = sample_latent_gaussian(
        prior.mean,
        prior.covariance,
        args.latent_draws,
        seed=args.seed + 1,
    )

    latents = clamp_latents(
        latents
    )

    coarse_count = min(
        args.coarse_draws,
        args.latent_draws,
    )

    fine_count = min(
        args.fine_draws,
        args.latent_draws,
    )

    coarse_latents = latents[
        :coarse_count
    ]

    fine_latents = latents[
        :fine_count
    ]

    # --------------------------------------------------------
    # Baselines
    # --------------------------------------------------------

    print(
        "Computing textbook baselines..."
    )

    baseline_coarse = (
        baseline_mean_win_rates(
            n=args.n,
            latents=coarse_latents,
            latent_chunk=
                args.latent_chunk,
        )
    )

    baseline_fine = (
        baseline_mean_win_rates(
            n=args.n,
            latents=fine_latents,
            latent_chunk=
                args.latent_chunk,
        )
    )

    baseline_full = (
        baseline_mean_win_rates(
            n=args.n,
            latents=latents,
            latent_chunk=
                args.latent_chunk,
        )
    )

    print(
        f"textbook modeled mean win rate = "
        f"{baseline_full.mean().item():.6f}"
    )

    # --------------------------------------------------------
    # Coarse search
    # --------------------------------------------------------

    gauges, chiralities = (
        coarse_candidates(
            args.gauge_points
        )
    )

    print()
    print(
        f"Coarse search: "
        f"{len(gauges)} candidates "
        f"x {coarse_count} latent draws"
    )

    coarse_records = evaluate_candidates(
        n=args.n,
        gauges=gauges,
        chiralities=chiralities,
        latents=coarse_latents,
        baseline=baseline_coarse,
        candidate_batch=
            args.candidate_batch,
        latent_chunk=
            args.latent_chunk,
    )

    coarse_top = top_records(
        coarse_records,
        metric=args.metric,
        count=args.coarse_keep,
    )

    print_top(
        "COARSE TOP",
        coarse_top,
        metric=args.metric,
    )

    # --------------------------------------------------------
    # Local refinement
    # --------------------------------------------------------

    refine_gauges, refine_chiralities = (
        refinement_candidates(
            coarse_top,
            coarse_points=
                args.gauge_points,
            refine_points=
                args.refine_points,
        )
    )

    print()
    print(
        f"Refinement: "
        f"{len(refine_gauges)} candidates "
        f"x {fine_count} latent draws"
    )

    refine_records = evaluate_candidates(
        n=args.n,
        gauges=refine_gauges,
        chiralities=
            refine_chiralities,
        latents=fine_latents,
        baseline=baseline_fine,
        candidate_batch=
            args.candidate_batch,
        latent_chunk=
            args.latent_chunk,
    )

    refine_top = top_records(
        refine_records,
        metric=args.metric,
        count=args.final_keep,
    )

    print_top(
        "REFINED TOP",
        refine_top,
        metric=args.metric,
    )

    # --------------------------------------------------------
    # Full Monte-Carlo evaluation of finalists
    # --------------------------------------------------------

    final_gauges = [
        float(
            record["gauge"]
        )
        for record in refine_top
    ]

    final_chiralities = [
        int(
            record["chirality"]
        )
        for record in refine_top
    ]

    # Always include exact textbook baseline.
    final_gauges.append(
        0.0
    )

    final_chiralities.append(
        +1
    )

    print()
    print(
        f"Final evaluation: "
        f"{len(final_gauges)} candidates "
        f"x {args.latent_draws} latent draws"
    )

    final_records = evaluate_candidates(
        n=args.n,
        gauges=final_gauges,
        chiralities=
            final_chiralities,
        latents=latents,
        baseline=baseline_full,
        candidate_batch=
            args.candidate_batch,
        latent_chunk=
            args.latent_chunk,
    )

    final_sorted = top_records(
        final_records,
        metric=args.metric,
        count=len(
            final_records
        ),
    )

    print_top(
        "FINALISTS",
        final_sorted,
        metric=args.metric,
        count=min(
            20,
            len(final_sorted),
        ),
    )

    winner = final_sorted[0]

    gauge = float(
        winner["gauge"]
    )

    chirality = int(
        winner["chirality"]
    )

    # --------------------------------------------------------
    # Verify exact noiseless value
    # --------------------------------------------------------

    ideal = ideal_win_rates(
        args.n,
        gauge,
        chirality,
    )

    ideal_mean = (
        sum(ideal)
        / len(ideal)
    )

    quantum_bound = omega_q(
        args.n
    )

    ideal_error = abs(
        ideal_mean
        - quantum_bound
    )

    if ideal_error > 1e-10:
        raise RuntimeError(
            "Selected equivalent strategy failed ideal verification: "
            f"{ideal_mean} vs {quantum_bound}"
        )

    # --------------------------------------------------------
    # CNOT direction
    # --------------------------------------------------------

    #
    # IMPORTANT:
    #
    # Current Emerald latent model has no CNOT-direction parameter.
    # Therefore forward and reverse preparations receive exactly the
    # same modeled score.
    #
    direction = (
        args.preferred_direction
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "n":
            args.n,

        "selection_metric":
            args.metric,

        "gauge":
            gauge,

        "chirality":
            chirality,

        "preferred_cnot_direction":
            direction,

        "cnot_direction_identifiable":
            False,

        "cnot_direction_note": (
            "Current Emerald latent twin models the post-preparation "
            "state and measurement/compiler errors but has no "
            "direction-specific entangler latent. Forward and reverse "
            "CNOT preparations are therefore tied in this model."
        ),

        "ideal_mean_win_rate":
            ideal_mean,

        "quantum_bound":
            quantum_bound,

        "ideal_error":
            ideal_error,

        "modeled_textbook_mean":
            float(
                baseline_full
                .mean()
                .item()
            ),

        "winner":
            winner,

        "finalists":
            final_sorted,
    }

    (
        args.output_dir
        / "optimizer_result.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    for direction_name in (
        "forward",
        "reverse",
    ):
        (
            args.output_dir
            / (
                "build_circuits_"
                f"{direction_name}.py"
            )
        ).write_text(
            render_build_circuits(
                gauge=gauge,
                chirality=
                    chirality,
                direction=
                    direction_name,
            ),
            encoding="utf-8",
        )

    print()
    print(
        "WINNER"
    )
    print(
        "======"
    )

    print()
    print(
        f"gauge              = "
        f"{gauge:+.12f}"
    )

    print(
        f"chirality          = "
        f"{chirality:+d}"
    )

    print(
        f"q10 uplift         = "
        f"{float(winner['q10_uplift']):+.8f}"
    )

    print(
        f"median uplift      = "
        f"{float(winner['median_uplift']):+.8f}"
    )

    print(
        f"mean uplift        = "
        f"{float(winner['mean_uplift']):+.8f}"
    )

    print(
        f"P(beat textbook)   = "
        f"{float(winner['p_beat_textbook']):.4f}"
    )

    print(
        f"ideal win rate     = "
        f"{ideal_mean:.12f}"
    )

    print(
        f"omega_q            = "
        f"{quantum_bound:.12f}"
    )

    print()
    print(
        "CNOT direction:"
    )

    print(
        "  forward and reverse are "
        "indistinguishable in the current model."
    )

    print(
        f"  defaulting to {direction!r}."
    )

    print()
    print(
        "generated:"
    )

    print(
        f"  {args.output_dir}/"
        "optimizer_result.json"
    )

    print(
        f"  {args.output_dir}/"
        "build_circuits_forward.py"
    )

    print(
        f"  {args.output_dir}/"
        "build_circuits_reverse.py"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
