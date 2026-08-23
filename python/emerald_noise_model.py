from __future__ import annotations

"""Emerald-specific task model for Qupacabrathon 2026.

This module is intentionally *not* a full-process tomography model.  Emerald is
an IQM Crystal-54 superconducting QPU whose native gate set is PRX + CZ.  The
challenge circuit is only two qubits and ultimately asks for computational-basis
outcomes after two local Y-plane measurement rotations.  We therefore use a
rich physical prior to generate plausible Emerald executions, then project that
prior into a compact observable twin that models exactly the three moments that
fully determine the four returned bit probabilities:

    m_A(a) = E[Z_A]
    m_B(b) = E[Z_B]
    C(a,b) = E[Z_A Z_B]

For bits z(0)=+1, z(1)=-1,

    P(i,j) = 1/4 * [1 + z_i m_A + z_j m_B + z_i z_j C].

The compact twin is 13-dimensional and uses all four count bins.  This is much
more sample-efficient than trying to fit every microscopic gate error from a
single two-qubit sweep.

Published/device anchors used as prior centers:
  * Emerald median 1Q fidelity: 99.93% (AWS launch characterization)
  * Emerald median 2Q fidelity: 99.5%  (AWS launch characterization)
  * Challenge-repository Emerald readout probe: e0~0.0013, e1~0.028
  * Challenge-repository price: $0.30/task + $0.00160/shot

Everything beyond those anchors is a conservative modeling choice, not a claim
that Emerald has the sampled error on a particular day or qubit pair.
"""

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Emerald facts / event pricing
# ---------------------------------------------------------------------------

EMERALD_TASK_DOLLARS = 0.30
EMERALD_SHOT_DOLLARS = 0.00160
EMERALD_READOUT_E0 = 0.0013
EMERALD_READOUT_E1 = 0.0280
EMERALD_MEDIAN_1Q_FIDELITY = 0.9993
EMERALD_MEDIAN_2Q_FIDELITY = 0.9950

P_3SIGMA = 0.0013498980316300946

# ---------------------------------------------------------------------------
# Rich Emerald physical prior
# ---------------------------------------------------------------------------

PHYSICAL_NAMES = (
    "bell_visibility",      # white-noise Bell visibility
    "amp_a",               # amplitude damping probability after Bell prep
    "amp_b",
    "dephase",             # |00><11| coherence loss fraction
    "prep_ry_a",           # coherent local Y bias in prepared Bell state
    "prep_ry_b",
    "bell_phase",          # relative phase between |00> and |11>
    "measurement_rel",     # A/B relative additive RY error [rad]
    "scale_a",             # folded-RY fractional scale error
    "scale_b",
    "yerr_a",              # error on inserted Y half-turn [rad]
    "yerr_b",
    "e0_a", "e1_a",       # readout 0->1 and 1->0, Alice wire
    "e0_b", "e1_b",       # readout 0->1 and 1->0, Bob wire
)
PHYSICAL_DIM = len(PHYSICAL_NAMES)


@dataclass(frozen=True)
class Backend:
    device: torch.device
    dtype: torch.dtype

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"


def choose_backend(device: str = "auto", dtype: str = "float32") -> Backend:
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    dtypes = {"float32": torch.float32, "float64": torch.float64}
    if dtype not in dtypes:
        raise ValueError("dtype must be float32 or float64")
    return Backend(dev, dtypes[dtype])


def backend_report(backend: Backend) -> str:
    lines = [f"torch={torch.__version__}", f"device={backend.device}", f"search_dtype={backend.dtype}"]
    if backend.is_cuda:
        idx = backend.device.index or 0
        props = torch.cuda.get_device_properties(idx)
        lines += [
            f"gpu={torch.cuda.get_device_name(idx)}",
            f"vram={props.total_memory/(1024**3):.1f} GiB",
            f"compute_capability={props.major}.{props.minor}",
        ]
    return "\n".join(lines)


def omega_c(n: int) -> float:
    return 1.0 - 1.0 / (2.0 * n)


def omega_q(n: int) -> float:
    return math.cos(math.pi / (4.0 * n)) ** 2


def emerald_cost(n: int, shots: int, twirls: int = 1) -> float:
    return 2.0 * n * (EMERALD_TASK_DOLLARS * twirls + EMERALD_SHOT_DOLLARS * shots)


def shots_for_budget(n: int, budget: float, twirls: int = 1) -> int:
    return math.floor((budget / (2.0 * n) - EMERALD_TASK_DOLLARS * twirls) / EMERALD_SHOT_DOLLARS)


def _trunc_normal(
    center: float,
    sigma: float,
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    lo: float,
    hi: float,
) -> torch.Tensor:
    x = center + sigma * torch.randn(shape, generator=generator, device=device, dtype=dtype)
    return x.clamp(lo, hi)


def sample_emerald_physical_prior(
    count: int,
    *,
    backend: Backend,
    seed: int = 20260822,
    drift_scale: float = 1.0,
) -> torch.Tensor:
    """Draw plausible *per-run* Emerald latent parameters.

    The widths intentionally cover more variation than the launch medians imply.
    They are priors for robust decision-making, not measured calibration data.
    Readout draws have a common component plus per-wire variation because both
    qubits share the same device/calibration epoch.
    """
    g = torch.Generator(device=backend.device)
    g.manual_seed(seed)
    dev, dt = backend.device, backend.dtype

    # State / entangler quality.  A 99.5% median 2Q gate fidelity does not map
    # one-to-one to Bell visibility, so we deliberately use a wider prior.
    visibility = _trunc_normal(0.995, 0.0040 * drift_scale, (count,), generator=g, device=dev, dtype=dt, lo=0.965, hi=0.999999)
    amp_a = _trunc_normal(0.0015, 0.0015 * drift_scale, (count,), generator=g, device=dev, dtype=dt, lo=0.0, hi=0.015)
    amp_b = _trunc_normal(0.0015, 0.0015 * drift_scale, (count,), generator=g, device=dev, dtype=dt, lo=0.0, hi=0.015)
    dephase = _trunc_normal(0.0030, 0.0030 * drift_scale, (count,), generator=g, device=dev, dtype=dt, lo=0.0, hi=0.03)

    # Coherent prep/measurement errors.  1Q fidelity 99.93% corresponds to a
    # small-angle coherent-error scale of order few*1e-2 rad if coherence were
    # the only source; these priors are intentionally broader.
    prep_ry_a = _trunc_normal(0.0, 0.020 * drift_scale, (count,), generator=g, device=dev, dtype=dt, lo=-0.12, hi=0.12)
    prep_ry_b = _trunc_normal(0.0, 0.020 * drift_scale, (count,), generator=g, device=dev, dtype=dt, lo=-0.12, hi=0.12)
    bell_phase = _trunc_normal(0.0, 0.035 * drift_scale, (count,), generator=g, device=dev, dtype=dt, lo=-0.20, hi=0.20)
    measurement_rel = _trunc_normal(0.0, 0.040 * drift_scale, (count,), generator=g, device=dev, dtype=dt, lo=-0.20, hi=0.20)
    scale_a = _trunc_normal(0.0, 0.012 * drift_scale, (count,), generator=g, device=dev, dtype=dt, lo=-0.06, hi=0.06)
    scale_b = _trunc_normal(0.0, 0.012 * drift_scale, (count,), generator=g, device=dev, dtype=dt, lo=-0.06, hi=0.06)
    yerr_a = _trunc_normal(0.0, 0.025 * drift_scale, (count,), generator=g, device=dev, dtype=dt, lo=-0.15, hi=0.15)
    yerr_b = _trunc_normal(0.0, 0.025 * drift_scale, (count,), generator=g, device=dev, dtype=dt, lo=-0.15, hi=0.15)

    # Readout: challenge-repository probe centers, with correlated day-to-day
    # motion and smaller wire-specific offsets.
    common_e0 = _trunc_normal(EMERALD_READOUT_E0, 0.0025 * drift_scale, (count,), generator=g, device=dev, dtype=dt, lo=1e-6, hi=0.030)
    common_e1 = _trunc_normal(EMERALD_READOUT_E1, 0.0070 * drift_scale, (count,), generator=g, device=dev, dtype=dt, lo=0.002, hi=0.080)
    e0_a = (common_e0 + 0.0015 * torch.randn((count,), generator=g, device=dev, dtype=dt)).clamp(1e-6, 0.04)
    e0_b = (common_e0 + 0.0015 * torch.randn((count,), generator=g, device=dev, dtype=dt)).clamp(1e-6, 0.04)
    e1_a = (common_e1 + 0.0040 * torch.randn((count,), generator=g, device=dev, dtype=dt)).clamp(1e-6, 0.10)
    e1_b = (common_e1 + 0.0040 * torch.randn((count,), generator=g, device=dev, dtype=dt)).clamp(1e-6, 0.10)

    return torch.stack(
        (
            visibility, amp_a, amp_b, dephase,
            prep_ry_a, prep_ry_b, bell_phase,
            measurement_rel, scale_a, scale_b,
            yerr_a, yerr_b,
            e0_a, e1_a, e0_b, e1_b,
        ),
        dim=-1,
    )


# ---------------------------------------------------------------------------
# Physical prior -> X/Z observable moments
# ---------------------------------------------------------------------------


def _ry(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta / 2.0)
    s = torch.sin(theta / 2.0)
    out = torch.zeros((*theta.shape, 2, 2), device=theta.device, dtype=torch.complex64 if theta.dtype == torch.float32 else torch.complex128)
    out[..., 0, 0] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    out[..., 1, 1] = c
    return out


def _kron2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Batched Kronecker product [...,2,2] x [...,2,2] -> [...,4,4]
    return torch.einsum("...ij,...kl->...ikjl", a, b).reshape(*a.shape[:-2], 4, 4)


def _apply_local_amplitude_damping(rho: torch.Tensor, ga: torch.Tensor, gb: torch.Tensor) -> torch.Tensor:
    cdtype = rho.dtype
    dev = rho.device
    one_a = torch.sqrt((1.0 - ga).clamp_min(0))
    one_b = torch.sqrt((1.0 - gb).clamp_min(0))
    sqrt_a = torch.sqrt(ga.clamp_min(0))
    sqrt_b = torch.sqrt(gb.clamp_min(0))

    def ks(one, sq):
        k0 = torch.zeros((*one.shape, 2, 2), device=dev, dtype=cdtype)
        k1 = torch.zeros_like(k0)
        k0[..., 0, 0] = 1.0
        k0[..., 1, 1] = one
        k1[..., 0, 1] = sq
        return k0, k1

    a0, a1 = ks(one_a, sqrt_a)
    b0, b1 = ks(one_b, sqrt_b)
    result = torch.zeros_like(rho)
    for ka in (a0, a1):
        for kb in (b0, b1):
            k = _kron2(ka, kb)
            result = result + k @ rho @ k.conj().transpose(-1, -2)
    return result


def physical_state_moments(params: torch.Tensor) -> torch.Tensor:
    """Return [D,8] = rAx,rAz,rBx,rBz,Txx,Txz,Tzx,Tzz before readout.

    The state begins as |Phi+>, receives white-noise visibility, Bell-coherence
    loss/phase, amplitude damping, then small local Y preparation errors.
    """
    if params.ndim == 1:
        params = params.unsqueeze(0)
    real_dtype = params.dtype
    cdtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
    dev = params.device
    d = params.shape[0]

    v, ga, gb, dephase = (params[:, i] for i in range(4))
    pa, pb, phase = (params[:, i] for i in (4, 5, 6))

    rho = torch.zeros((d, 4, 4), device=dev, dtype=cdtype)
    rho[:, 0, 0] = 0.5
    rho[:, 3, 3] = 0.5
    coherence = 0.5 * (1.0 - dephase) * torch.exp(-1j * phase.to(cdtype))
    rho[:, 0, 3] = coherence
    rho[:, 3, 0] = coherence.conj()
    eye = torch.eye(4, device=dev, dtype=cdtype).unsqueeze(0)
    rho = v[:, None, None] * rho + (1.0 - v)[:, None, None] * eye / 4.0

    rho = _apply_local_amplitude_damping(rho, ga, gb)
    ua = _ry(pa)
    ub = _ry(pb)
    u = _kron2(ua, ub)
    rho = u @ rho @ u.conj().transpose(-1, -2)

    X = torch.tensor([[0, 1], [1, 0]], device=dev, dtype=cdtype)
    Z = torch.tensor([[1, 0], [0, -1]], device=dev, dtype=cdtype)
    I = torch.eye(2, device=dev, dtype=cdtype)
    ops = torch.stack(
        (
            torch.kron(X, I), torch.kron(Z, I),
            torch.kron(I, X), torch.kron(I, Z),
            torch.kron(X, X), torch.kron(X, Z),
            torch.kron(Z, X), torch.kron(Z, Z),
        ),
        dim=0,
    )
    moments = torch.einsum("dij,kji->dk", rho, ops).real.to(real_dtype)
    return moments


def fold_ry_tensor(angle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    turns = torch.round(angle / math.pi)
    rem = angle - turns * math.pi
    parity = torch.remainder(turns.to(torch.int64), 2).to(angle.dtype)
    return rem, parity


def physical_observed_moments(
    a_requested: torch.Tensor,
    b_requested: torch.Tensor,
    params: torch.Tensor,
    state_moments: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Observed mA,mB,C for arbitrary requested angle pairs.

    a_requested,b_requested are [G]. params is [D,16]. Returns [D,G] each.
    """
    if params.ndim == 1:
        params = params.unsqueeze(0)
    if state_moments is None:
        state_moments = physical_state_moments(params)
    a = a_requested.to(device=params.device, dtype=params.dtype).reshape(1, -1)
    b = b_requested.to(device=params.device, dtype=params.dtype).reshape(1, -1)
    ar, ap = fold_ry_tensor(a)
    br, bp = fold_ry_tensor(b)

    rel = params[:, 7:8]
    sa, sb = params[:, 8:9], params[:, 9:10]
    ya, yb = params[:, 10:11], params[:, 11:12]
    aeff = (1.0 + sa) * ar + ap * (math.pi + ya) + 0.5 * rel
    beff = (1.0 + sb) * br + bp * (math.pi + yb) - 0.5 * rel
    sina, cosa = torch.sin(aeff), torch.cos(aeff)
    sinb, cosb = torch.sin(beff), torch.cos(beff)

    rax, raz, rbx, rbz, txx, txz, tzx, tzz = [state_moments[:, i:i+1] for i in range(8)]
    ma = rax * sina + raz * cosa
    mb = rbx * sinb + rbz * cosb
    corr = txx * sina * sinb + txz * sina * cosb + tzx * cosa * sinb + tzz * cosa * cosb

    e0a, e1a, e0b, e1b = [params[:, i:i+1] for i in range(12, 16)]
    alpha_a, beta_a = e1a - e0a, 1.0 - e0a - e1a
    alpha_b, beta_b = e1b - e0b, 1.0 - e0b - e1b
    ma_obs = alpha_a + beta_a * ma
    mb_obs = alpha_b + beta_b * mb
    c_obs = alpha_a * alpha_b + alpha_a * beta_b * mb + alpha_b * beta_a * ma + beta_a * beta_b * corr
    return ma_obs, mb_obs, c_obs


# ---------------------------------------------------------------------------
# 17-dimensional compiler-aware Emerald latent twin
# ---------------------------------------------------------------------------

LATENT_NAMES = (
    "rAx", "rAz", "rBx", "rBz", "Txx", "Txz", "Tzx", "Tzz",
    "measurement_rel", "scale_a", "scale_b", "yerr_a", "yerr_b",
    "e0_a", "e1_a", "e0_b", "e1_b",
)
LATENT_DIM = len(LATENT_NAMES)


@dataclass(frozen=True)
class EmeraldLatentPrior:
    mean: torch.Tensor
    covariance: torch.Tensor
    physical_draws: int


def physical_to_latent(params: torch.Tensor) -> torch.Tensor:
    """Project microscopic Emerald prior samples into exactly observable latents.

    The first eight entries are the Bell-state X/Z moments after state-prep
    channels. The remaining nine entries are compiler-aware measurement and
    readout parameters. No harmonic approximation is used.
    """
    state = physical_state_moments(params)
    return torch.cat((state, params[:, 7:16]), dim=-1)


def build_emerald_latent_prior(
    *,
    backend: Backend,
    physical_draws: int = 262_144,
    seed: int = 20260822,
    covariance_inflate: float = 1.15,
) -> EmeraldLatentPrior:
    params = sample_emerald_physical_prior(physical_draws, backend=backend, seed=seed)
    latent = physical_to_latent(params)
    mean = latent.mean(dim=0)
    centered = latent - mean
    covariance = centered.T @ centered / max(1, physical_draws - 1)
    covariance = covariance * (covariance_inflate ** 2)
    # Tiny discrepancy floor keeps the prior nonsingular and acknowledges
    # unmodeled compiler/pulse drift without swamping the Emerald anchors.
    floor = torch.tensor(
        [
            0.0015,0.0020,0.0015,0.0020,
            0.0025,0.0020,0.0020,0.0025,
            0.0060,0.0030,0.0030,0.0060,0.0060,
            0.0008,0.0020,0.0008,0.0020,
        ], device=backend.device, dtype=backend.dtype,
    )
    covariance = covariance + torch.diag(floor.square())
    return EmeraldLatentPrior(mean, covariance, physical_draws)


def latent_observed_moments(
    a_requested: torch.Tensor,
    b_requested: torch.Tensor,
    latent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Observed mA,mB,C from the compiler-aware 17D latent twin.

    a_requested,b_requested: broadcast-compatible [...]
    latent: [17] or [D,17]
    returns matching moments, with D prepended for batched latents.
    """
    single = latent.ndim == 1
    if single:
        latent = latent.unsqueeze(0)
    dt, dev = latent.dtype, latent.device
    a = a_requested.to(device=dev, dtype=dt)
    b = b_requested.to(device=dev, dtype=dt)
    # Put latent batch in front of arbitrary angle dimensions.
    shape = (latent.shape[0],) + (1,) * a.ndim
    def lv(i): return latent[:, i].reshape(shape)

    ar, ap = fold_ry_tensor(a)
    br, bp = fold_ry_tensor(b)
    ar = ar.unsqueeze(0); ap = ap.unsqueeze(0)
    br = br.unsqueeze(0); bp = bp.unsqueeze(0)

    rel, sa, sb, ya, yb = [lv(i) for i in range(8,13)]
    aeff = (1.0 + sa) * ar + ap * (math.pi + ya) + 0.5 * rel
    beff = (1.0 + sb) * br + bp * (math.pi + yb) - 0.5 * rel
    sina,cosa = torch.sin(aeff),torch.cos(aeff)
    sinb,cosb = torch.sin(beff),torch.cos(beff)

    rax,raz,rbx,rbz,txx,txz,tzx,tzz = [lv(i) for i in range(8)]
    ma = rax*sina + raz*cosa
    mb = rbx*sinb + rbz*cosb
    corr = txx*sina*sinb + txz*sina*cosb + tzx*cosa*sinb + tzz*cosa*cosb

    e0a,e1a,e0b,e1b = [lv(i) for i in range(13,17)]
    aa,ba = e1a-e0a, 1.0-e0a-e1a
    ab,bb = e1b-e0b, 1.0-e0b-e1b
    ma_obs = aa + ba*ma
    mb_obs = ab + bb*mb
    c_obs = aa*ab + aa*bb*mb + ab*ba*ma + ba*bb*corr
    if single:
        return ma_obs[0],mb_obs[0],c_obs[0]
    return ma_obs,mb_obs,c_obs


def probabilities_from_moments(ma: torch.Tensor, mb: torch.Tensor, cc: torch.Tensor, *, clamp: bool = True) -> torch.Tensor:
    p00=0.25*(1+ma+mb+cc)
    p01=0.25*(1+ma-mb-cc)
    p10=0.25*(1-ma+mb-cc)
    p11=0.25*(1-ma-mb+cc)
    p=torch.stack((p00,p01,p10,p11),dim=-1)
    if clamp:
        p=p.clamp_min(1e-8)
        p=p/p.sum(dim=-1,keepdim=True)
    return p


def latent_probabilities(a: torch.Tensor,b: torch.Tensor,latent: torch.Tensor,*,clamp: bool=True)->torch.Tensor:
    return probabilities_from_moments(*latent_observed_moments(a,b,latent),clamp=clamp)


def question_order(n:int,*,device:torch.device)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    idx=torch.arange(n,device=device,dtype=torch.long)
    x=torch.cat((idx,idx)); y=torch.cat((idx,(idx+1)%n))
    vertex=torch.cat((torch.ones(n,device=device,dtype=torch.bool),torch.zeros(n,device=device,dtype=torch.bool)))
    return x,y,vertex


def question_probabilities_from_tables(n:int,alice:torch.Tensor,bob:torch.Tensor,latent:torch.Tensor,*,clamp:bool=True)->torch.Tensor:
    if alice.ndim==1:
        alice=alice.unsqueeze(0); bob=bob.unsqueeze(0)
    x,y,_=question_order(n,device=alice.device)
    a=alice[:,x]; b=bob[:,y]
    p=latent_probabilities(a,b,latent,clamp=clamp)
    # latent batch, if present, precedes design batch.
    return p


def question_win_rates(n:int,probs:torch.Tensor)->torch.Tensor:
    _,_,vertex=question_order(n,device=probs.device)
    same=probs[...,0]+probs[...,3]; diff=probs[...,1]+probs[...,2]
    shape=[1]*(same.ndim-1)+[same.shape[-1]]
    return torch.where(vertex.reshape(shape),same,diff)


def textbook_angles(n:int,*,device:torch.device,dtype:torch.dtype)->tuple[torch.Tensor,torch.Tensor]:
    theta=math.pi/(4.0*n); step=math.pi-4.0*theta
    idx=torch.arange(n,device=device,dtype=dtype)
    return idx*step, idx*step+2.0*theta


# ---------------------------------------------------------------------------
# End of single-run model
# ---------------------------------------------------------------------------

# The earlier two-stage planner carried Fisher-information and posterior-update
# helpers below this point.  The single-run workflow deliberately has no such
# stage: it evaluates strategies directly against the Emerald prior ensemble.

