#!/usr/bin/env python3
"""The pre-submission gate. Every hardware run passes this before it is queued.

    uv run --python 3.12 scripts/plan_check.py --n 13 --shots 53 --delta 0.015

Answers three questions about a run plan and nothing else. Does it certify at
the deficit the team declared, with the power the published tables are sized
for, does it fit what the team has left against its cap, and how much deficit
margin does it keep? The margin is the largest deficit the plan still
certifies: run a device worse than that and the plan returns a loss no matter
how clean the code is. `--balance` defaults to the full per-team cap in
dollars, so pass it to check a second run against what the first one left.

`--qpu` picks which QPU the plan is priced on, and defaults to the event's
default device. It changes the price and nothing else: both QPUs charge the
same task fee, and Garnet's per-shot rate is about 10% below Emerald's, so a
shot-heavy sweep is cheaper there and a task-heavy one costs the same. Pass the
device the submission actually names, because pricing a Garnet run at Emerald's
rate overstates it by up to about $1.60 inside this cap and costs the team reach
it had.

`--delta` is required and ships no default, because the event publishes no
device deficit. It is the team's own number, reached by predicting it from the
published device facts in the background notes or by measuring it with a run,
and this tool checks only that the plan is internally consistent with it. A
declared deficit better than the device's truth is a bet: the gate here will
pass and the hardware run will fail to certify. That is priced rather than
warned about, and it is the same money either way.

The point of a machine gate is that a mentor cannot read twelve run plans in one
afternoon, and a plan that fails here fails silently on hardware after the money
is gone. Since every team spends from one shared pot, a plan that overspends costs
the other eleven teams too, so run this against the parameters in the submitted
script rather than a plan typed separately. Passing here is necessary, never
sufficient. A mentor still sees the script, and an organizer still runs it.

Exit status is 0 when the plan passes both checks and 1 when it fails either, so
this can gate a submission script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from game_numbers import (  # noqa: E402
    CAP_PER_TEAM,
    DEFAULT_QPU,
    MINIMUM_SHOTS,
    P_3SIGMA,
    POWER_TARGET,
    QPU_ROUTES,
    certification_power,
    cost,
    critical_visibility,
    largest_delta_certified,
    largest_n_affordable,
    n_circuits,
    omega_c,
    omega_q,
    p_value,
    quantum_advantage,
    rates_for_qpu,
    shots_for_significance,
)


def check(
    n: int,
    shots: int,
    delta: float,
    balance: float = CAP_PER_TEAM,
    twirls: int = 1,
    qpu: str = DEFAULT_QPU,
) -> int:
    if qpu not in QPU_ROUTES:
        print(
            f"FAIL  qpu = {qpu!r} is not a device this event offers. Pick one of "
            f"{sorted(QPU_ROUTES)}."
        )
        return 1
    if not 0.0 <= delta < 0.5:
        print(f"FAIL  delta = {delta} is not a device deficit in [0, 0.5).")
        return 1
    if n < 3 or n % 2 == 0:
        print(f"FAIL  n = {n} is not an odd cycle. Pick an odd n >= 3.")
        return 1
    if shots < MINIMUM_SHOTS:
        print(
            f"FAIL  shots = {shots} is below the {MINIMUM_SHOTS} floor the event "
            "enforces."
        )
        return 1
    if not 1 <= twirls <= 16:
        print(f"FAIL  twirls = {twirls} is not between 1 and 16.")
        return 1
    if shots % twirls != 0:
        print(
            f"FAIL  shots = {shots} does not split into twirls = {twirls} equal "
            "variants. Pick shots as a multiple of twirls; the S_90 tables are "
            "sized at twirls = 1, so round S up to the next multiple."
        )
        return 1
    if shots // twirls < MINIMUM_SHOTS:
        print(
            f"FAIL  {shots} shots over twirls = {twirls} variants is "
            f"{shots // twirls} shots per variant, below the {MINIMUM_SHOTS} "
            "floor the event enforces per circuit task."
        )
        return 1

    circuits = n_circuits(n)
    omega_expected = omega_q(n) - delta
    eps = omega_expected - omega_c(n)
    price = cost(n, shots, twirls, qpu)
    task_rate, shot_rate = rates_for_qpu(qpu)

    print(f"Run plan: C_{n}, {shots} shots per circuit, {circuits} circuits")
    print(
        f"  device            {qpu}, at ${task_rate:.2f} per task + "
        f"${shot_rate:.5f} per shot"
    )
    if twirls > 1:
        print(
            f"  twirled: {twirls} variants per question at {shots // twirls} "
            f"shots each, {twirls} task fees per question"
        )
    print(f"  classical bound   omega_c = {omega_c(n):.6f}")
    print(f"  quantum bound     omega_q = {omega_q(n):.6f}")
    print(f"  margin to spend             {quantum_advantage(n):.6f}")
    print(f"  min device visibility       {critical_visibility(n):.4f}")
    print(f"  declared deficit  delta   = {delta:.4f}  [your number, not the event's]")
    print(f"  expected win rate omega   = {omega_expected:.6f}")
    print()

    # --- Gate 1: does the plan certify? -----------------------------------
    if eps <= 0:
        print(f"FAIL  gate: at delta = {delta:.4f} the device cannot beat omega_c at n = {n}.")
        print(f"            Expected win rate is {abs(eps):.6f} below the classical bound.")
        print("            No shot count repairs this. Lower n, or lower delta with mitigation.")
        gate_ok = False
    else:
        # eps > 0 here, so shots_for_significance always returns a count.
        sized = shots_for_significance(n, delta)
        assert sized is not None
        power = certification_power(n, shots, delta)
        expected_p = p_value([omega_expected] * circuits, shots, omega_c(n))
        gate_ok = expected_p <= P_3SIGMA
        verdict = "PASS" if gate_ok else "FAIL"
        print(
            f"{verdict}  gate: this plan certifies with probability {power:.2f} "
            f"at delta = {delta:.4f}"
        )
        print(
            f"            expected outcome p = {expected_p:.3e} against the "
            f"3 sigma threshold {P_3SIGMA:.3e}"
        )
        print(
            f"            sized for {POWER_TARGET:.0%} power this n needs {sized} "
            f"shots per circuit (${cost(n, sized, qpu=qpu):.2f}), plan has {shots}"
        )
        if not gate_ok:
            print(
                f"            short by {sized - shots} shots per circuit against "
                "the published table"
            )
        elif power < POWER_TARGET:
            print(
                f"            below the {POWER_TARGET:.0%} target: this plan is a "
                f"gamble that fails {100 * (1 - power):.0f}% of the time even at "
                "the delta you declared"
            )

    # --- Gate 2: does the plan fit the balance? ---------------------------
    print()
    per_circuit = price / circuits
    cost_ok = price <= balance
    verdict = "PASS" if cost_ok else "FAIL"
    print(f"{verdict}  cost: ${price:.2f} against a balance of ${balance:.2f}")
    print(f"            {circuits} circuits at ${per_circuit:.4f} each")
    if cost_ok:
        print(f"            ${balance - price:.2f} left over")
    else:
        print(f"            over by ${price - balance:.2f}")
        affordable = largest_n_affordable(balance, delta, qpu=qpu)
        if affordable is None:
            print(
                f"            no odd n is certifiable within ${balance:.2f} at "
                "this delta"
            )
        else:
            alt_n, alt_shots, alt_price = affordable
            print(
                f"            largest n this balance certifies at "
                f"{POWER_TARGET:.0%} power: n = {alt_n} at {alt_shots} shots per "
                f"circuit, ${alt_price:.2f}"
            )

    # --- The team's own margin -------------------------------------------
    # The meter is continuous, so there are no free boundary shots to point at:
    # every shot above the sized plan is a purchase, and what it buys is margin.
    print()
    margin = largest_delta_certified(n, shots)
    print(
        f"Margin: this plan keeps {POWER_TARGET:.0%} certification probability "
        f"up to delta = {margin:.4f}."
    )
    if margin <= 0:
        print("        That is no margin at all. The plan is under-shot at every device quality.")
    elif margin < delta:
        print(f"        Below the declared {delta:.4f}, so the plan is under-shot.")
    else:
        print(f"        {margin - delta:+.4f} of headroom against the declared {delta:.4f}.")
        print("        Nothing here checks that declared number against the device.")
        print("        A cheap low-n run measures it, and that measurement is yours.")

    print()
    passed = gate_ok and cost_ok
    print("PLAN PASSES" if passed else "PLAN FAILS")
    if passed:
        print("Next: run scripts/submission_check.py against the script itself, then")
        print("open the pull request. A mentor approves it and an organizer executes.")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check a Qupacabrathon hardware run plan before it is submitted.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", type=int, required=True, help="odd cycle size, odd and >= 3")
    parser.add_argument("--shots", type=int, required=True, help="shots per circuit")
    parser.add_argument(
        "--balance",
        type=float,
        default=CAP_PER_TEAM,
        help="dollars still available to this team against its cap across both runs",
    )
    parser.add_argument(
        "--delta",
        type=float,
        required=True,
        help=(
            "the device deficit omega_q - omega you are sizing against. Required, "
            "and there is no default: predict it from the published device facts "
            "in the background notes, or measure it with a run"
        ),
    )
    parser.add_argument(
        "--qpu",
        default=DEFAULT_QPU,
        choices=sorted(QPU_ROUTES),
        help=(
            "which QPU this run goes to. It changes the price and nothing else: "
            "both devices charge the same task fee and Garnet's per-shot rate is "
            "about 10%% below Emerald's"
        ),
    )
    parser.add_argument(
        "--twirls",
        type=int,
        default=1,
        help="Pauli-twirled variants per question; each is a separately billed task",
    )
    args = parser.parse_args()
    return check(args.n, args.shots, args.delta, args.balance, args.twirls, args.qpu)


if __name__ == "__main__":
    raise SystemExit(main())
