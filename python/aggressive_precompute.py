from __future__ import annotations

"""Aggressive offline strategy compiler for Qupacabrathon 2026.

The expensive work is classical and happens here, preferably on an RTX 5090.
The emitted build_circuits() contains only precomputed angle tables and the
four-gate odd-cycle circuit used by the challenge.

Two modes:

1. run1
   Search for an information-rich calibration strategy under a prior over QPU
   errors, while enforcing robust certification constraints.

2. run2
   Consume run1_fit.json, sample its Laplace posterior, and search for the
   largest odd cycle that can robustly certify inside the remaining budget.

The GPU stage uses FP32/FP64 batched tensor algebra and a normal approximation
for screening. Every selected winner is re-evaluated on CPU/float64 with an
exact total-win distribution at the posterior/prior center before it is emitted.
"""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Iterable

import numpy as np
from scipy.stats import binom
import torch

from gpu_backend import (
    PARAMETER_NAMES,
    backend_report,
    choose_backend,
    cost,
    nominal_params_tensor,
    omega_c,
    prior_sigma_tensor,
    sample_prior,
    shots_for_budget,
)

P_3SIGMA = 0.0013498980316300946


# ---------------------------------------------------------------------------
# Strategy families
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Family:
    name: str
    size: int
    labels: tuple[str, ...]
    default_radius: tuple[float, ...]


FAMILIES: dict[str, Family] = {
    # c0 is always a relative A/B offset: +c0/2 on Alice, -c0/2 on Bob.
    "affine": Family(
        "affine",
        3,
        ("relative_offset", "alice_linear", "bob_linear"),
        (0.45, 0.30, 0.30),
    ),
    "quadratic": Family(
        "quadratic",
        5,
        (
            "relative_offset",
            "alice_linear",
            "alice_quadratic",
            "bob_linear",
            "bob_quadratic",
        ),
        (0.45, 0.30, 0.20, 0.30, 0.20),
    ),
    # Linear + first Fourier harmonic. 7 parameters total.
    "fourier1": Family(
        "fourier1",
        7,
        (
            "relative_offset",
            "alice_linear",
            "alice_sin1",
            "alice_cos1",
            "bob_linear",
            "bob_sin1",
            "bob_cos1",
        ),
        (0.45, 0.25, 0.18, 0.18, 0.25, 0.18, 0.18),
    ),
}


def _base_angles(n: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    theta = math.pi / (4.0 * n)
    step = math.pi - 4.0 * theta
    idx = torch.arange(n, device=device, dtype=dtype)
    return idx * step, idx * step + 2.0 * theta


def coefficients_to_angles(
    n: int,
    coefficients: torch.Tensor,
    family_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert strategy coefficients to explicit legal Alice/Bob angle tables.

    coefficients: [B,K] or [K]
    returns A, B: [B,n]

    Every Alice correction is a function only of x, and every Bob correction
    only of y, preserving the nonlocal game's locality constraint.
    """
    family = FAMILIES[family_name]
    if coefficients.ndim == 1:
        coefficients = coefficients.unsqueeze(0)
    if coefficients.shape[-1] != family.size:
        raise ValueError(
            f"{family_name} expects {family.size} coefficients, got {coefficients.shape[-1]}"
        )

    device, dtype = coefficients.device, coefficients.dtype
    a0, b0 = _base_angles(n, device=device, dtype=dtype)
    u = torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype)
    # Centered quadratic so it is less correlated with the constant term.
    q = u.square() - u.square().mean()
    phase = 2.0 * math.pi * torch.arange(n, device=device, dtype=dtype) / n
    s1, c1 = torch.sin(phase), torch.cos(phase)

    rel = coefficients[:, 0:1]
    if family_name == "affine":
        ca = coefficients[:, 1:2] * u
        cb = coefficients[:, 2:3] * u
    elif family_name == "quadratic":
        ca = coefficients[:, 1:2] * u + coefficients[:, 2:3] * q
        cb = coefficients[:, 3:4] * u + coefficients[:, 4:5] * q
    elif family_name == "fourier1":
        ca = (
            coefficients[:, 1:2] * u
            + coefficients[:, 2:3] * s1
            + coefficients[:, 3:4] * c1
        )
        cb = (
            coefficients[:, 4:5] * u
            + coefficients[:, 5:6] * s1
            + coefficients[:, 6:7] * c1
        )
    else:
        raise ValueError(f"unknown family {family_name!r}")

    return a0.unsqueeze(0) + 0.5 * rel + ca, b0.unsqueeze(0) - 0.5 * rel + cb


def sample_coefficients(
    count: int,
    family_name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    radius_scale: float,
    include_textbook: bool,
) -> torch.Tensor:
    family = FAMILIES[family_name]
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    radius = torch.tensor(
        family.default_radius,
        device=device,
        dtype=dtype,
    ) * radius_scale

    # A mixture improves coverage: 70% approximately Gaussian around textbook,
    # 30% uniform across the full box for aggressive exploration.
    gaussian_count = int(round(0.70 * count))
    uniform_count = count - gaussian_count
    gaussian = torch.randn(
        (gaussian_count, family.size), generator=gen, device=device, dtype=dtype
    ) * (0.35 * radius)
    uniform = (
        2.0 * torch.rand(
            (uniform_count, family.size), generator=gen, device=device, dtype=dtype
        )
        - 1.0
    ) * radius
    out = torch.cat((gaussian, uniform), dim=0)
    out = torch.max(torch.min(out, radius), -radius)
    if include_textbook and count:
        out[0].zero_()
    return out


# ---------------------------------------------------------------------------
# Device execution model (vectorized explicit angle tables)
# ---------------------------------------------------------------------------


def question_xy(n: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.cat((torch.arange(n), torch.arange(n))).to(device=device, dtype=torch.long)
    y = torch.cat((torch.arange(n), (torch.arange(n) + 1) % n)).to(device=device, dtype=torch.long)
    return x, y, x == y


def _fold_ry(angle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    turns = torch.round(angle / math.pi)
    remainder = angle - turns * math.pi
    parity = torch.remainder(turns.to(torch.int64), 2).to(angle.dtype)
    return remainder, parity


def sweep_probabilities_angles(
    n: int,
    alice_angles: torch.Tensor,
    bob_angles: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    """Probabilities for explicit angle tables.

    A/B: [B,n], params: [D,8] or [8]
    returns: [B,D,2n,4] ordered 00,01,10,11.
    """
    if alice_angles.ndim == 1:
        alice_angles = alice_angles.unsqueeze(0)
    if bob_angles.ndim == 1:
        bob_angles = bob_angles.unsqueeze(0)
    if params.ndim == 1:
        params = params.unsqueeze(0)
    if alice_angles.shape != bob_angles.shape:
        raise ValueError("Alice and Bob angle tables must have equal shape")
    if alice_angles.device != params.device:
        raise ValueError("angles and params must be on the same device")
    params = params.to(dtype=alice_angles.dtype)

    x, y, _ = question_xy(n, device=alice_angles.device)
    a_req = alice_angles[:, x]
    b_req = bob_angles[:, y]
    a_rem, a_parity = _fold_ry(a_req)
    b_rem, b_parity = _fold_ry(b_req)

    rel, scale_a, scale_b, yerr_a, yerr_b, visibility, e0, e1 = params.T
    a_eff = (
        (1.0 + scale_a[None, :, None]) * a_rem[:, None, :]
        + a_parity[:, None, :] * (math.pi + yerr_a[None, :, None])
        + 0.5 * rel[None, :, None]
    )
    b_eff = (
        (1.0 + scale_b[None, :, None]) * b_rem[:, None, :]
        + b_parity[:, None, :] * (math.pi + yerr_b[None, :, None])
        - 0.5 * rel[None, :, None]
    )

    delta = a_eff - b_eff
    agree = torch.cos(0.5 * delta).square()
    v = visibility[None, :, None]
    p_same_each = 0.5 * v * agree + 0.25 * (1.0 - v)
    p_diff_each = 0.5 * v * (1.0 - agree) + 0.25 * (1.0 - v)
    true = torch.stack((p_same_each, p_diff_each, p_diff_each, p_same_each), dim=-1)

    t00, t01, t10, t11 = true.unbind(-1)
    c00 = (1.0 - e0)[None, :, None]
    c01 = e1[None, :, None]
    c10 = e0[None, :, None]
    c11 = (1.0 - e1)[None, :, None]
    o00 = c00 * (c00 * t00 + c01 * t01) + c01 * (c00 * t10 + c01 * t11)
    o01 = c00 * (c10 * t00 + c11 * t01) + c01 * (c10 * t10 + c11 * t11)
    o10 = c10 * (c00 * t00 + c01 * t01) + c11 * (c00 * t10 + c01 * t11)
    o11 = c10 * (c10 * t00 + c11 * t01) + c11 * (c10 * t10 + c11 * t11)
    out = torch.stack((o00, o01, o10, o11), dim=-1)
    return out / out.sum(dim=-1, keepdim=True)


def question_win_rates(n: int, probs: torch.Tensor) -> torch.Tensor:
    _, _, vertex = question_xy(n, device=probs.device)
    same = probs[..., 0] + probs[..., 3]
    diff = probs[..., 1] + probs[..., 2]
    shape = [1] * (same.ndim - 1) + [same.shape[-1]]
    return torch.where(vertex.reshape(shape), same, diff)


def critical_win_count(n: int, shots: int) -> int:
    total = 2 * n * shots
    q = omega_c(n)
    lo, hi = 0, total
    while lo < hi:
        mid = (lo + hi) // 2
        if float(binom.sf(mid - 1, total, q)) <= P_3SIGMA:
            hi = mid
        else:
            lo = mid + 1
    return lo


def approximate_power(rates: torch.Tensor, shots: int, critical_wins: int) -> torch.Tensor:
    mean = shots * rates.sum(dim=-1)
    variance = shots * (rates * (1.0 - rates)).sum(dim=-1)
    z = (critical_wins - 0.5 - mean) / variance.clamp_min(1e-12).sqrt()
    return 0.5 * torch.erfc(z / math.sqrt(2.0))


def fisher_information_gain(
    n: int,
    shots: int,
    alice: torch.Tensor,
    bob: torch.Tensor,
    center: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Laplace/Fisher information gain for candidate strategies at one device."""
    if center.ndim != 1:
        raise ValueError("center must have shape [8]")
    dtype, device = alice.dtype, alice.device
    p0 = sweep_probabilities_angles(n, alice, bob, center)[:, 0]
    steps = torch.tensor(
        [2e-4, 2e-4, 2e-4, 2e-4, 2e-4, 2e-5, 2e-5, 2e-5],
        device=device,
        dtype=dtype,
    )
    eye = torch.eye(8, device=device, dtype=dtype)
    plus = center[None, :] + eye * steps[:, None]
    minus = center[None, :] - eye * steps[:, None]
    plus[:, 5] = plus[:, 5].clamp(0.850001, 0.999999)
    minus[:, 5] = minus[:, 5].clamp(0.850001, 0.999999)
    plus[:, 6:] = plus[:, 6:].clamp(1e-7, 0.119999)
    minus[:, 6:] = minus[:, 6:].clamp(1e-7, 0.119999)
    denom = (plus - minus).diagonal().clone()
    p_plus = sweep_probabilities_angles(n, alice, bob, plus)
    p_minus = sweep_probabilities_angles(n, alice, bob, minus)
    jac = (p_plus - p_minus) / denom[None, :, None, None]
    inv_p = p0.clamp_min(1e-8).reciprocal()
    fisher = shots * torch.einsum("bpqo,brqo,bqo->bpr", jac, jac, inv_p)
    scaled = fisher * sigma[None, :, None] * sigma[None, None, :]
    mat = torch.eye(8, device=device, dtype=dtype).unsqueeze(0) + scaled
    sign, logdet = torch.linalg.slogdet(mat)
    return torch.where(sign > 0, 0.5 * logdet, torch.full_like(logdet, -torch.inf))


# ---------------------------------------------------------------------------
# Priors / posteriors
# ---------------------------------------------------------------------------


def _clamp_device_draws(draws: torch.Tensor) -> torch.Tensor:
    draws[:, 0] = draws[:, 0].clamp(-0.35, 0.35)
    draws[:, 1:3] = draws[:, 1:3].clamp(-0.12, 0.12)
    draws[:, 3:5] = draws[:, 3:5].clamp(-0.25, 0.25)
    draws[:, 5] = draws[:, 5].clamp(0.85, 0.999999)
    draws[:, 6] = draws[:, 6].clamp(1e-7, 0.10)
    draws[:, 7] = draws[:, 7].clamp(1e-7, 0.12)
    return draws


def load_fit(path: Path) -> tuple[np.ndarray, np.ndarray]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    center = np.array([doc["map"][name] for name in PARAMETER_NAMES], dtype=float)
    sigma = np.array([doc["laplaceStd"][name] for name in PARAMETER_NAMES], dtype=float)
    sigma = np.maximum(sigma, np.array([1e-4] * 8))
    return center, sigma


def sample_device_distribution(
    mode: str,
    qpu: str,
    count: int,
    *,
    backend,
    seed: int,
    fit_path: Path | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (center, sigma, draws)."""
    if mode == "run1":
        center = nominal_params_tensor(qpu, device=backend.device, dtype=backend.search_dtype)
        sigma = prior_sigma_tensor(device=backend.device, dtype=backend.search_dtype)
        draws = sample_prior(
            qpu,
            count,
            device=backend.device,
            dtype=backend.search_dtype,
            seed=seed,
        )
        return center, sigma, draws

    if fit_path is None:
        raise ValueError("run2 mode requires --fit run1_fit.json")
    center_np, sigma_np = load_fit(fit_path)
    center = torch.tensor(center_np, device=backend.device, dtype=backend.search_dtype)
    sigma = torch.tensor(sigma_np, device=backend.device, dtype=backend.search_dtype)
    gen = torch.Generator(device=backend.device)
    gen.manual_seed(seed)
    draws = center[None, :] + torch.randn(
        (count, 8), generator=gen, device=backend.device, dtype=backend.search_dtype
    ) * sigma[None, :]
    return center, sigma, _clamp_device_draws(draws)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    family: str
    n: int
    shots: int
    coefficients: np.ndarray
    nominal_info: float
    nominal_power: float
    robust_median_power: float = float("nan")
    robust_p10_power: float = float("nan")
    robust_p01_power: float = float("nan")
    robust_success90: float = float("nan")
    robust_mean_win: float = float("nan")
    score: float = float("-inf")


def _push_top(
    top: list[Candidate],
    new_rows: Iterable[Candidate],
    limit: int,
    *,
    mode: str,
) -> list[Candidate]:
    top.extend(new_rows)
    if mode == "run1":
        top.sort(key=lambda c: (c.score, c.nominal_info, c.nominal_power), reverse=True)
    else:
        top.sort(key=lambda c: (c.score, c.nominal_power), reverse=True)
    return top[:limit]


def screen_family(
    *,
    mode: str,
    family_name: str,
    n: int,
    shots: int,
    center: torch.Tensor,
    sigma: torch.Tensor,
    candidates: int,
    batch_size: int,
    radius_scale: float,
    nominal_power_floor: float,
    keep: int,
    seed: int,
    complexity_penalty: float,
    backend,
) -> list[Candidate]:
    family = FAMILIES[family_name]
    threshold = critical_win_count(n, shots)
    top: list[Candidate] = []
    processed = 0
    while processed < candidates:
        count = min(batch_size, candidates - processed)
        coeff = sample_coefficients(
            count,
            family_name,
            device=backend.device,
            dtype=backend.search_dtype,
            seed=seed + processed * 17,
            radius_scale=radius_scale,
            include_textbook=(processed == 0),
        )
        a, b = coefficients_to_angles(n, coeff, family_name)
        probs = sweep_probabilities_angles(n, a, b, center)[:, 0]
        rates = question_win_rates(n, probs)
        power = approximate_power(rates, shots, threshold)

        if mode == "run1":
            info = fisher_information_gain(n, shots, a, b, center, sigma)
            # Complexity penalty is deliberately small; robustness is the main guard.
            score = info - complexity_penalty * family.size
        else:
            info = torch.zeros_like(power)
            # Final-run screening prioritizes certification probability directly.
            score = power - complexity_penalty * family.size

        score = torch.where(
            power >= nominal_power_floor,
            score,
            torch.full_like(score, -torch.inf),
        )
        k = min(keep, count)
        values, indices = torch.topk(score, k=k)
        rows: list[Candidate] = []
        for val, idx in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
            if not math.isfinite(val):
                continue
            rows.append(
                Candidate(
                    family=family_name,
                    n=n,
                    shots=shots,
                    coefficients=coeff[idx].detach().cpu().double().numpy(),
                    nominal_info=float(info[idx].detach().cpu()),
                    nominal_power=float(power[idx].detach().cpu()),
                    score=float(val),
                )
            )
        top = _push_top(top, rows, keep, mode=mode)
        processed += count

    # Never allow an aggressive search to forget the textbook strategy.  The
    # robust stage must explicitly compare every fancy family against zero
    # correction, even when the latter has lower nominal information gain.
    zero = torch.zeros((1, family.size), device=backend.device, dtype=backend.search_dtype)
    a0, b0 = coefficients_to_angles(n, zero, family_name)
    probs0 = sweep_probabilities_angles(n, a0, b0, center)[:, 0]
    rates0 = question_win_rates(n, probs0)
    power0 = float(approximate_power(rates0, shots, threshold)[0].detach().cpu())
    if mode == "run1":
        info0 = float(fisher_information_gain(n, shots, a0, b0, center, sigma)[0].detach().cpu())
        score0 = info0 - complexity_penalty * family.size
    else:
        info0 = 0.0
        score0 = power0 - complexity_penalty * family.size
    baseline = Candidate(
        family=family_name,
        n=n,
        shots=shots,
        coefficients=np.zeros(family.size, dtype=float),
        nominal_info=info0,
        nominal_power=power0,
        score=score0,
    )
    if not any(np.allclose(c.coefficients, 0.0, atol=1e-12) for c in top):
        top.append(baseline)
    return top


def robust_rerank(
    rows: list[Candidate],
    *,
    draws: torch.Tensor,
    candidate_chunk: int,
    device_chunk: int,
    complexity_penalty: float,
    mode: str,
    robust_power_floor: float,
    backend,
) -> list[Candidate]:
    """Evaluate finalists across device draws without materializing a huge tensor."""
    if not rows:
        return rows

    # Group by n/family/shots because tensors in one batch must have equal shape.
    groups: dict[tuple[str, int, int], list[Candidate]] = {}
    for row in rows:
        groups.setdefault((row.family, row.n, row.shots), []).append(row)

    for (family_name, n, shots), group in groups.items():
        threshold = critical_win_count(n, shots)
        for start in range(0, len(group), candidate_chunk):
            part = group[start : start + candidate_chunk]
            coeff = torch.tensor(
                np.stack([c.coefficients for c in part]),
                device=backend.device,
                dtype=backend.search_dtype,
            )
            a, b = coefficients_to_angles(n, coeff, family_name)
            power_pieces: list[torch.Tensor] = []
            win_pieces: list[torch.Tensor] = []
            for d0 in range(0, draws.shape[0], device_chunk):
                d = draws[d0 : d0 + device_chunk]
                probs = sweep_probabilities_angles(n, a, b, d)
                rates = question_win_rates(n, probs)
                power_pieces.append(approximate_power(rates, shots, threshold).detach().cpu())
                win_pieces.append(rates.mean(dim=-1).detach().cpu())
            powers = torch.cat(power_pieces, dim=1).numpy()  # C,D
            wins = torch.cat(win_pieces, dim=1).numpy()
            for i, c in enumerate(part):
                c.robust_median_power = float(np.median(powers[i]))
                c.robust_p10_power = float(np.percentile(powers[i], 10))
                c.robust_p01_power = float(np.percentile(powers[i], 1))
                c.robust_success90 = float(np.mean(powers[i] >= 0.90))
                c.robust_mean_win = float(np.mean(wins[i]))
                if mode == "run1":
                    # Reward information, but only after demanding robust survival.
                    survival = min(1.0, c.robust_p10_power / max(robust_power_floor, 1e-9))
                    c.score = (
                        c.nominal_info
                        * survival
                        * (0.50 + 0.50 * c.robust_success90)
                        - complexity_penalty * FAMILIES[c.family].size
                    )
                    if c.robust_p10_power < robust_power_floor:
                        c.score -= 100.0 * (robust_power_floor - c.robust_p10_power)
                else:
                    # For final runs the p10 and p01 tail dominate the score.
                    c.score = (
                        2.0 * c.robust_p10_power
                        + 0.75 * c.robust_p01_power
                        + 0.25 * c.robust_median_power
                        - complexity_penalty * FAMILIES[c.family].size
                    )
                    if c.robust_p10_power < robust_power_floor:
                        c.score -= 100.0 * (robust_power_floor - c.robust_p10_power)

    rows.sort(key=lambda c: c.score, reverse=True)
    return rows




def tune_run2_shots(
    rows: list[Candidate],
    *,
    draws: torch.Tensor,
    robust_power_floor: float,
    device_chunk: int,
    complexity_penalty: float,
    backend,
) -> list[Candidate]:
    """Reduce each finalist to the smallest shot count meeting the p10 floor.

    Angle probabilities do not depend on shot count, so we compute them once and
    then solve the integer shot-sizing problem classically. This avoids spending
    the entire remaining budget merely because it is available.
    """
    for c in rows:
        coeff = torch.tensor(
            c.coefficients[None, :], device=backend.device, dtype=backend.search_dtype
        )
        a, b = coefficients_to_angles(c.n, coeff, c.family)
        rate_parts: list[torch.Tensor] = []
        for d0 in range(0, draws.shape[0], device_chunk):
            probs = sweep_probabilities_angles(c.n, a, b, draws[d0:d0+device_chunk])[0]
            rate_parts.append(question_win_rates(c.n, probs).detach().cpu())
        rates = torch.cat(rate_parts, dim=0).double().numpy()  # D,Q

        def metrics(shots: int) -> tuple[float, float, float, float]:
            threshold = critical_win_count(c.n, shots)
            mean = shots * rates.sum(axis=-1)
            var = shots * np.sum(rates * (1.0 - rates), axis=-1)
            z = (threshold - 0.5 - mean) / np.sqrt(np.maximum(var, 1e-15))
            # Normal SF using erf; numpy does not expose erf on every build, so
            # torch on CPU gives a stable vectorized implementation.
            zt = torch.from_numpy(z)
            power = (0.5 * torch.erfc(zt / math.sqrt(2.0))).numpy()
            return (
                float(np.median(power)),
                float(np.percentile(power, 10)),
                float(np.percentile(power, 1)),
                float(np.mean(power >= 0.90)),
            )

        max_shots = c.shots
        max_metrics = metrics(max_shots)
        if max_metrics[1] < robust_power_floor:
            continue

        lo, hi = 10, max_shots
        while lo < hi:
            mid = (lo + hi) // 2
            if metrics(mid)[1] >= robust_power_floor:
                hi = mid
            else:
                lo = mid + 1
        # Exact threshold discreteness can create tiny local wiggles. Walk down
        # while the previous integer still meets the target.
        chosen = lo
        while chosen > 10 and metrics(chosen - 1)[1] >= robust_power_floor:
            chosen -= 1
        med, p10, p01, success90 = metrics(chosen)
        c.shots = chosen
        c.robust_median_power = med
        c.robust_p10_power = p10
        c.robust_p01_power = p01
        c.robust_success90 = success90
        c.score = (
            2.0 * p10 + 0.75 * p01 + 0.25 * med
            - complexity_penalty * FAMILIES[c.family].size
            - 1e-6 * chosen  # deterministic tie-break toward fewer shots
        )
    rows.sort(key=lambda c: (c.n, c.score, -c.shots), reverse=True)
    return rows

# ---------------------------------------------------------------------------
# Exact center validation and emitter
# ---------------------------------------------------------------------------


def _fold_ry_np(angle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    turns = np.round(angle / math.pi)
    return angle - turns * math.pi, np.remainder(turns.astype(np.int64), 2).astype(float)


def center_probabilities_np(
    n: int,
    alice: np.ndarray,
    bob: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    x = np.concatenate((np.arange(n), np.arange(n)))
    y = np.concatenate((np.arange(n), (np.arange(n) + 1) % n))
    a_req, b_req = alice[x], bob[y]
    a_rem, a_par = _fold_ry_np(a_req)
    b_rem, b_par = _fold_ry_np(b_req)
    rel, sa, sb, ya, yb, v, e0, e1 = params
    a_eff = (1 + sa) * a_rem + a_par * (math.pi + ya) + 0.5 * rel
    b_eff = (1 + sb) * b_rem + b_par * (math.pi + yb) - 0.5 * rel
    agree = np.cos(0.5 * (a_eff - b_eff)) ** 2
    ps = 0.5 * v * agree + 0.25 * (1 - v)
    pd = 0.5 * v * (1 - agree) + 0.25 * (1 - v)
    true = np.stack((ps, pd, pd, ps), axis=-1)
    C = np.array([[1 - e0, e1], [e0, 1 - e1]], dtype=float)
    out = []
    for row in true:
        m = C @ row.reshape(2, 2) @ C.T
        out.append(m.reshape(4) / m.sum())
    return np.asarray(out)


def exact_total_win_power(n: int, shots: int, rates: np.ndarray) -> float:
    """Exact distribution for total wins with each question repeated `shots` times."""
    threshold = critical_win_count(n, shots)
    dist = np.array([1.0])
    k = np.arange(shots + 1)
    for p in rates:
        pmf = binom.pmf(k, shots, float(p))
        dist = np.convolve(dist, pmf)
    return float(dist[threshold:].sum())


def candidate_angle_tables(row: Candidate, *, dtype=torch.float64) -> tuple[np.ndarray, np.ndarray]:
    coeff = torch.tensor(row.coefficients, dtype=dtype)
    a, b = coefficients_to_angles(row.n, coeff, row.family)
    return a[0].cpu().numpy(), b[0].cpu().numpy()


def exact_center_validation(row: Candidate, center: torch.Tensor) -> tuple[float, float]:
    a, b = candidate_angle_tables(row)
    p = center.detach().cpu().double().numpy()
    probs = center_probabilities_np(row.n, a, b, p)
    vertex = np.array([True] * row.n + [False] * row.n)
    rates = np.where(vertex, probs[:, 0] + probs[:, 3], probs[:, 1] + probs[:, 2])
    return float(rates.mean()), exact_total_win_power(row.n, row.shots, rates)


def emit_build_circuits(row: Candidate, output: Path) -> None:
    a, b = candidate_angle_tables(row)
    lines = [
        '# Generated by aggressive_precompute.py.\n',
        '# Paste this function (and only this function) into a copy of\n',
        '# challenge/submission-template.py. The template already imports qml and Callable.\n',
        f'# family={row.family}; N={row.n}; shots={row.shots}\n',
        '\n',
        'def build_circuits(n: int, theta: float) -> list[Callable[[], None]]:\n',
        f'    if n != {row.n}:\n',
        f'        raise ValueError(f"precomputed strategy is for C_{row.n}, got C_{{n}}")\n',
        '    # theta is intentionally unused: all requested angles were compiled offline.\n',
        '    alice_angles = (\n',
    ]
    for value in a:
        lines.append(f'        {float(value):+.17g},\n')
    lines += ['    )\n', '    bob_angles = (\n']
    for value in b:
        lines.append(f'        {float(value):+.17g},\n')
    lines += [
        '    )\n',
        '\n',
        '    def gates_for(x: int, y: int) -> Callable[[], None]:\n',
        '        angle_a = alice_angles[x]\n',
        '        angle_b = bob_angles[y]\n',
        '\n',
        '        def circuit() -> None:\n',
        '            qml.Hadamard(wires=0)\n',
        '            qml.CNOT(wires=[0, 1])\n',
        '            qml.RY(angle_a, wires=0)\n',
        '            qml.RY(angle_b, wires=1)\n',
        '\n',
        '        return circuit\n',
        '\n',
        '    return [gates_for(x, y) for x, y in question_order(n)]\n',
    ]
    output.write_text(''.join(lines), encoding='utf-8')


def save_report(
    output: Path,
    *,
    args,
    center: torch.Tensor,
    sigma: torch.Tensor,
    winners: list[Candidate],
) -> None:
    rows = []
    for c in winners:
        mean_center, exact_power = exact_center_validation(c, center)
        a, b = candidate_angle_tables(c)
        rows.append(
            {
                "family": c.family,
                "n": c.n,
                "shotsPerQuestion": c.shots,
                "cost": cost(c.n, c.shots, args.qpu),
                "coefficients": {
                    label: float(value)
                    for label, value in zip(FAMILIES[c.family].labels, c.coefficients, strict=True)
                },
                "nominalInfoNats": c.nominal_info,
                "nominalApproxPower": c.nominal_power,
                "robustMedianPower": c.robust_median_power,
                "robustP10Power": c.robust_p10_power,
                "robustP01Power": c.robust_p01_power,
                "fractionDevicesPowerAtLeast90": c.robust_success90,
                "robustMeanWin": c.robust_mean_win,
                "score": c.score,
                "exactCenterMeanWin": mean_center,
                "exactCenterCertificationPower": exact_power,
                "aliceAngles": [float(x) for x in a],
                "bobAngles": [float(x) for x in b],
            }
        )
    doc = {
        "schemaVersion": 1,
        "mode": args.mode,
        "qpu": args.qpu,
        "budget": args.budget,
        "fit": None if args.fit is None else str(args.fit),
        "deviceParameterCenter": {
            name: float(v)
            for name, v in zip(PARAMETER_NAMES, center.detach().cpu().double().tolist(), strict=True)
        },
        "deviceParameterSigma": {
            name: float(v)
            for name, v in zip(PARAMETER_NAMES, sigma.detach().cpu().double().tolist(), strict=True)
        },
        "winners": rows,
    }
    output.write_text(json.dumps(doc, indent=2) + '\n', encoding='utf-8')


def parse_n_values(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo, hi = map(int, part.split('-', 1))
            values.extend(range(lo, hi + 1, 2))
        else:
            values.append(int(part))
    values = sorted(set(values))
    for n in values:
        if n < 3 or n % 2 == 0:
            raise ValueError(f'N must be odd and >=3, got {n}')
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description='Aggressive RTX-GPU Qupacabrathon strategy compiler')
    parser.add_argument('--mode', choices=('run1', 'run2'), default='run1')
    parser.add_argument('--qpu', choices=('emerald', 'garnet'), default='emerald')
    parser.add_argument('--budget', type=float, default=3.54, help='Budget for this run only')
    parser.add_argument('--fit', type=Path, help='run1_fit.json; required for --mode run2')
    parser.add_argument('--n', default='3,5', help='Comma list or odd range, e.g. 3,5 or 9-19')
    parser.add_argument('--families', nargs='+', choices=tuple(FAMILIES), default=['affine','quadratic','fourier1'])
    parser.add_argument('--candidates', type=int, default=1_000_000, help='Candidates per family per N')
    parser.add_argument('--batch-size', type=int, default=16_384)
    parser.add_argument('--screen-keep', type=int, default=256, help='Keep per family/N after nominal screening')
    parser.add_argument('--robust-keep', type=int, default=32, help='Top candidates to report after robust reranking')
    parser.add_argument('--device-draws', type=int, default=32_768)
    parser.add_argument('--device-chunk', type=int, default=4096)
    parser.add_argument('--candidate-chunk', type=int, default=32)
    parser.add_argument('--radius-scale', type=float, default=1.0)
    parser.add_argument('--nominal-power-floor', type=float, default=0.90)
    parser.add_argument('--robust-p10-floor', type=float, default=0.80)
    parser.add_argument('--complexity-penalty', type=float, default=0.004)
    parser.add_argument('--seed', type=int, default=20260822)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--dtype', choices=('float32','float64'), default='float32')
    parser.add_argument('--out-dir', type=Path, default=Path('generated_aggressive'))
    args = parser.parse_args()

    if args.mode == 'run2' and args.fit is None:
        parser.error('--mode run2 requires --fit run1_fit.json')

    n_values = parse_n_values(args.n)
    backend = choose_backend(args.device, args.dtype)
    print('Aggressive precomputation backend')
    print(backend_report(backend))
    print(f'mode={args.mode}, qpu={args.qpu}, budget=${args.budget:.2f}')
    print(f'N={n_values}; families={args.families}')
    print()

    center, sigma, draws = sample_device_distribution(
        args.mode,
        args.qpu,
        args.device_draws,
        backend=backend,
        seed=args.seed + 77,
        fit_path=args.fit,
    )

    finalists: list[Candidate] = []
    t_all = time.perf_counter()
    for n in n_values:
        shots = shots_for_budget(n, args.budget, args.qpu)
        if shots < 10:
            print(f'C_{n}: skip; budget yields only {shots} shots/question')
            continue
        print(f'C_{n}: {shots} shots/question, cost=${cost(n, shots, args.qpu):.3f}')
        for family_name in args.families:
            t0 = time.perf_counter()
            top = screen_family(
                mode=args.mode,
                family_name=family_name,
                n=n,
                shots=shots,
                center=center,
                sigma=sigma,
                candidates=args.candidates,
                batch_size=args.batch_size,
                radius_scale=args.radius_scale,
                nominal_power_floor=args.nominal_power_floor,
                keep=args.screen_keep,
                seed=args.seed + n * 1009 + FAMILIES[family_name].size * 7919,
                complexity_penalty=args.complexity_penalty,
                backend=backend,
            )
            finalists.extend(top)
            best = top[0] if top else None
            if best is None:
                print(f'  {family_name:<10} no nominal candidate passes power floor')
            else:
                print(
                    f'  {family_name:<10} screened {args.candidates:,}; '
                    f'best info={best.nominal_info:.4f}, power≈{best.nominal_power:.2%}; '
                    f'{time.perf_counter()-t0:.2f}s'
                )
        print()

    if not finalists:
        raise SystemExit('No nominally feasible candidates found.')

    print(f'Robust rerank: {len(finalists):,} finalists x {args.device_draws:,} device draws')
    t0 = time.perf_counter()
    finalists = robust_rerank(
        finalists,
        draws=draws,
        candidate_chunk=args.candidate_chunk,
        device_chunk=args.device_chunk,
        complexity_penalty=args.complexity_penalty,
        mode=args.mode,
        robust_power_floor=args.robust_p10_floor,
        backend=backend,
    )
    print(f'robust rerank finished in {time.perf_counter()-t0:.2f}s\n')

    # In run2, prefer the largest N that clears the robust floor. This prevents a
    # tiny C3 with near-certain success from outranking the actual competition axis.
    if args.mode == 'run2':
        feasible = [c for c in finalists if c.robust_p10_power >= args.robust_p10_floor]
        if feasible:
            max_n = max(c.n for c in feasible)
            selected_pool = [c for c in feasible if c.n == max_n]
            selected_pool.sort(key=lambda c: c.score, reverse=True)
            winners = selected_pool[: args.robust_keep]
            winners = tune_run2_shots(
                winners,
                draws=draws,
                robust_power_floor=args.robust_p10_floor,
                device_chunk=args.device_chunk,
                complexity_penalty=args.complexity_penalty,
                backend=backend,
            )
        else:
            finalists.sort(key=lambda c: (c.n, c.score), reverse=True)
            winners = finalists[: args.robust_keep]
    else:
        winners = finalists[: args.robust_keep]

    print('Top robust candidates')
    for rank, c in enumerate(winners[:10], 1):
        center_win, center_exact = exact_center_validation(c, center)
        coeff_text = ', '.join(f'{x:+.5f}' for x in c.coefficients)
        print(
            f'#{rank:>2} C_{c.n:<2} {c.family:<10} '
            f'p10={c.robust_p10_power:6.2%} p01={c.robust_p01_power:6.2%} '
            f'median={c.robust_median_power:6.2%} P[p>=90]={c.robust_success90:6.2%} '
            f'info={c.nominal_info:6.3f} exact-center={center_exact:6.2%}'
        )
        print(f'     coeff=[{coeff_text}], center_win={center_win:.6f}')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / f'{args.mode}_search_report.json'
    build_path = args.out_dir / f'{args.mode}_build_circuits.py'
    save_report(report_path, args=args, center=center, sigma=sigma, winners=winners)
    emit_build_circuits(winners[0], build_path)
    print(f'\nwrote {report_path}')
    print(f'wrote {build_path}')
    print(f'total wall time: {time.perf_counter()-t_all:.2f}s')


if __name__ == '__main__':
    main()

