from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from noise_model import DEFAULT_PRIOR_SIGMA, QPU_READOUT, QPU_RATES

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
class Backend:
    device: torch.device
    search_dtype: torch.dtype
    validation_dtype: torch.dtype

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"


def choose_backend(device: str = "auto", search_dtype: str = "float32") -> Backend:
    if device == "auto":
        selected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        selected = torch.device(device)
        if selected.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")

    dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
    }.get(search_dtype)
    if dtype is None:
        raise ValueError("search_dtype must be 'float32' or 'float64'")
    return Backend(selected, dtype, torch.float64)


def backend_report(backend: Backend) -> str:
    lines = [
        f"torch={torch.__version__}",
        f"device={backend.device}",
        f"search_dtype={backend.search_dtype}",
    ]
    if backend.is_cuda:
        index = backend.device.index or 0
        props = torch.cuda.get_device_properties(index)
        lines.extend(
            [
                f"gpu={torch.cuda.get_device_name(index)}",
                f"vram={props.total_memory / (1024**3):.1f} GiB",
                f"compute_capability={props.major}.{props.minor}",
            ]
        )
    return "\n".join(lines)


def nominal_params_tensor(qpu: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if qpu not in QPU_READOUT:
        raise ValueError(f"unknown QPU {qpu!r}")
    e0, e1 = QPU_READOUT[qpu]
    return torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.995, e0, e1],
        device=device,
        dtype=dtype,
    )


def prior_sigma_tensor(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(DEFAULT_PRIOR_SIGMA, device=device, dtype=dtype)


def omega_c(n: int) -> float:
    return 1.0 - 1.0 / (2.0 * n)


def omega_q(n: int) -> float:
    return math.cos(math.pi / (4.0 * n)) ** 2


def shots_for_budget(n: int, budget: float, qpu: str, twirls: int = 1) -> int:
    task, shot = QPU_RATES[qpu]
    return math.floor((budget / (2.0 * n) - task * twirls) / shot)


def cost(n: int, shots: int, qpu: str, twirls: int = 1) -> float:
    task, shot = QPU_RATES[qpu]
    return 2.0 * n * (task * twirls + shot * shots)


def question_xy(n: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.cat((torch.arange(n), torch.arange(n))).to(device=device, dtype=torch.long)
    y = torch.cat((torch.arange(n), (torch.arange(n) + 1) % n)).to(device=device, dtype=torch.long)
    is_vertex = x == y
    return x, y, is_vertex


def base_angles(n: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    theta = math.pi / (4.0 * n)
    step = math.pi - 4.0 * theta
    idx = torch.arange(n, device=device, dtype=dtype)
    a = idx * step
    b = idx * step + 2.0 * theta
    return a, b


def requested_angles_cross(
    n: int,
    probes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return requested angles for a batch of legal probe designs.

    probes: [B, 3] = [relative_offset, ramp_a, ramp_b]
    returns A, B each [B, n]
    """
    if probes.ndim == 1:
        probes = probes.unsqueeze(0)
    if probes.shape[-1] != 3:
        raise ValueError("probes must have shape [B, 3]")
    device, dtype = probes.device, probes.dtype
    a0, b0 = base_angles(n, device=device, dtype=dtype)
    u = torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype)
    offset = probes[:, 0:1]
    a = a0.unsqueeze(0) + 0.5 * offset + probes[:, 1:2] * u.unsqueeze(0)
    b = b0.unsqueeze(0) - 0.5 * offset + probes[:, 2:3] * u.unsqueeze(0)
    return a, b


def _fold_ry_tensor(angle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Mirrors Python math.remainder(angle, pi) closely: subtract the nearest
    # integer multiple of pi. Exact half-integer ties are irrelevant for the
    # angles used here and torch.round follows round-to-even.
    turns = torch.round(angle / math.pi)
    remainder = angle - turns * math.pi
    parity = torch.remainder(turns.to(torch.int64), 2).to(angle.dtype)
    return remainder, parity


def sweep_probabilities_cross(
    n: int,
    probes: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    """Vectorized probability model over all probes x all device parameters.

    probes: [B, 3]
    params: [D, 8] or [8]
    returns: [B, D, 2n, 4] in answer order 00,01,10,11.
    """
    if probes.ndim == 1:
        probes = probes.unsqueeze(0)
    if params.ndim == 1:
        params = params.unsqueeze(0)
    if probes.device != params.device:
        raise ValueError("probes and params must be on the same device")
    if probes.dtype != params.dtype:
        params = params.to(dtype=probes.dtype)

    device, dtype = probes.device, probes.dtype
    a_req, b_req = requested_angles_cross(n, probes)  # B,n
    x, y, _ = question_xy(n, device=device)
    a_q = a_req[:, x]  # B,Q
    b_q = b_req[:, y]

    a_rem, a_parity = _fold_ry_tensor(a_q)
    b_rem, b_parity = _fold_ry_tensor(b_q)

    # Broadcast B probes against D possible devices.
    p = params
    rel = p[:, 0]
    scale_a = p[:, 1]
    scale_b = p[:, 2]
    yerr_a = p[:, 3]
    yerr_b = p[:, 4]
    visibility = p[:, 5]
    e0 = p[:, 6]
    e1 = p[:, 7]

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

    true = torch.stack(
        (p_same_each, p_diff_each, p_diff_each, p_same_each), dim=-1
    )  # B,D,Q,4

    # Apply C @ P @ C^T analytically, avoiding tiny matrix multiplies.
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
    """probs [...,Q,4] -> rates [...,Q]."""
    _, _, is_vertex = question_xy(n, device=probs.device)
    same = probs[..., 0] + probs[..., 3]
    diff = probs[..., 1] + probs[..., 2]
    shape = [1] * (same.ndim - 1) + [same.shape[-1]]
    mask = is_vertex.reshape(shape)
    return torch.where(mask, same, diff)


def finite_difference_fisher_batch(
    n: int,
    shots: int,
    probes: torch.Tensor,
    params: torch.Tensor,
    prior_sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fisher info and approximate information gain for B probes at one device.

    params must be shape [8]. This is the high-throughput nominal-search kernel.
    Returns fisher [B,8,8], info_nats [B].
    """
    if params.ndim != 1 or params.numel() != 8:
        raise ValueError("params must have shape [8]")
    center = params
    dtype, device = probes.dtype, probes.device
    p0 = sweep_probabilities_cross(n, probes, center)[..., 0, :, :]  # B,Q,4

    # Parameter-specific finite difference steps. The computations are batched
    # so eight parameter derivatives cost one large GPU launch group.
    steps = torch.tensor(
        [2e-4, 2e-4, 2e-4, 2e-4, 2e-4, 2e-5, 2e-5, 2e-5],
        device=device,
        dtype=dtype,
    )
    eye = torch.eye(8, device=device, dtype=dtype)
    plus = center[None, :] + eye * steps[:, None]
    minus = center[None, :] - eye * steps[:, None]
    # Keep probability-like values inside their physical domain.
    plus[:, 5] = plus[:, 5].clamp(0.850001, 0.999999)
    minus[:, 5] = minus[:, 5].clamp(0.850001, 0.999999)
    plus[:, 6:] = plus[:, 6:].clamp(1e-7, 0.119999)
    minus[:, 6:] = minus[:, 6:].clamp(1e-7, 0.119999)
    denom = (plus - minus).diagonal().clone()  # [8]

    p_plus = sweep_probabilities_cross(n, probes, plus)   # B,8,Q,4
    p_minus = sweep_probabilities_cross(n, probes, minus)
    jac = (p_plus - p_minus) / denom[None, :, None, None]  # B,P,Q,O

    inv_p = p0.clamp_min(1e-8).reciprocal()
    fisher = shots * torch.einsum("bpqo,brqo,bqo->bpr", jac, jac, inv_p)

    scaled = fisher * prior_sigma[None, :, None] * prior_sigma[None, None, :]
    matrix = torch.eye(8, device=device, dtype=dtype).unsqueeze(0) + scaled
    sign, logabsdet = torch.linalg.slogdet(matrix)
    info = torch.where(sign > 0, 0.5 * logabsdet, torch.full_like(logabsdet, -torch.inf))
    return fisher, info


def approximate_certification_power_batch(
    n: int,
    shots: int,
    rates: torch.Tensor,
    critical_wins: int,
) -> torch.Tensor:
    """Normal approximation to Poisson-binomial power, vectorized."""
    mean = shots * rates.sum(dim=-1)
    variance = shots * (rates * (1.0 - rates)).sum(dim=-1)
    z = (critical_wins - 0.5 - mean) / variance.clamp_min(1e-12).sqrt()
    # sf(z) == Phi(-z)
    return 0.5 * torch.erfc(z / math.sqrt(2.0))


def sample_prior(
    qpu: str,
    count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    center = nominal_params_tensor(qpu, device=device, dtype=dtype)
    sigma = prior_sigma_tensor(device=device, dtype=dtype)
    draws = center[None, :] + torch.randn((count, 8), generator=gen, device=device, dtype=dtype) * sigma
    draws[:, 0] = draws[:, 0].clamp(-0.25, 0.25)
    draws[:, 1:3] = draws[:, 1:3].clamp(-0.10, 0.10)
    draws[:, 3:5] = draws[:, 3:5].clamp(-0.20, 0.20)
    draws[:, 5] = draws[:, 5].clamp(0.90, 0.999999)
    draws[:, 6] = draws[:, 6].clamp(1e-7, 0.08)
    draws[:, 7] = draws[:, 7].clamp(1e-7, 0.10)
    return draws

