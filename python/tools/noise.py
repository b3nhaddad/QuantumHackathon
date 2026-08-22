import math
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import pennylane as qml

# solution.py and game_numbers.py are siblings of this file's parent
# (python/), not of tools/, so they aren't importable by name until that
# directory is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game_numbers import GATE_2Q_ERROR, PUBLISHED_READOUT, omega_q
from solution import N, SHOTS, build_circuits, is_win, question_order, strategy_angle

def build_mathmatical(n: int, theta: float) -> list[Callable[[], None]]:
    # theta = pi / (4n)
    #
    # The optimal odd-cycle strategy uses:
    #
    #   Alice(x) = x * (pi - pi/n)
    #   Bob(y)   = y * (pi - pi/n) + pi/(2n)
    #
    # Since the template gives us theta:
    #
    #   pi/n    = 4 theta
    #   pi/(2n) = 2 theta

    step = math.pi - 4.0 * theta

    def alice_angle(x: int) -> float:
        return x * step

    def bob_angle(y: int) -> float:
        return y * step + 2.0 * theta

    def gates_for(x: int, y: int) -> Callable[[], None]:
        angle_a = alice_angle(x)
        angle_b = bob_angle(y)

        def circuit() -> None:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])

            qml.RY(angle_a, wires=0)
            qml.RY(angle_b, wires=1)

        return circuit

    return [
        gates_for(x, y)
        for x, y in question_order(n)
    ]


# Noise needs density matrices, not statevectors, so this device is always
# "default.mixed" regardless of what solution.py's own DEVICE parameter says:
# that parameter picks where the noiseless submission runs (simulator or
# qbraid hardware), which is a different question from what simulates a
# noise channel here. N and SHOTS are still solution's, so a sweep size or
# shot budget can never drift between the two files.
def noisy_device(wires: int):
    return qml.device("default.mixed", wires=wires)


_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def two_qubit_depolarizing_kraus(p: float) -> list[np.ndarray]:
    """Kraus operators for the standard 2-qubit depolarizing channel.

    rho -> (1 - p) rho + p * I/4: with probability p the two-qubit state is
    replaced by the maximally mixed state, otherwise it is untouched. Built
    from all 16 two-qubit Pauli strings P_i = P_a (x) P_b, using the identity
    (1/16) sum_i P_i rho P_i^dagger = I/4 for the 16-element Pauli basis on two
    qubits. That gives weight p/16 on each of the 15 non-identity strings and
    weight (1 - p) + p/16 = 1 - 15p/16 on the identity string, which is a
    proper CPTP map (weights sum to 1, and it reduces to no-op at p = 0).
    """
    labels = [a + b for a in "IXYZ" for b in "IXYZ"]
    kraus = []
    for label in labels:
        weight = (1.0 - p) + p / 16.0 if label == "II" else p / 16.0
        op = np.kron(_PAULI[label[0]], _PAULI[label[1]])
        kraus.append(math.sqrt(weight) * op)
    return kraus


def two_qubit_gate_error(p: float, wires):
    """The circuit's one CNOT, run through a 2-qubit depolarizing channel.

    `game_numbers.GATE_2Q_ERROR` is the median CZ/CNOT gate error published
    for both offered devices (99.5% fidelity). The channel is unitarily
    invariant -- conjugating (1-p) rho + p I/4 by any subsequent unitary on
    the same two wires gives back the same channel applied to the rotated
    state -- so applying it once, anywhere between the CNOT and the
    measurement, is exact for this circuit (CNOT then two single-qubit RYs),
    not an approximation of splicing it in immediately after the gate.
    """
    qml.QubitChannel(two_qubit_depolarizing_kraus(p), wires=list(wires))


def asymmetric_bit_flip(e0: float, e1: float, wires):
    """Readout error that reads a true 0 as 1 w.p. e0 and a true 1 as 0 w.p. e1.

    Kraus form of the channel `game_numbers.delta_from_readout_asym` derives
    its formulas for: K0 leaves each basis state alone with its own survival
    amplitude, K1 carries |0> to |1> at rate e0, K2 carries |1> to |0> at rate
    e1. Reduces to a symmetric bit flip of probability p on each wire (a
    `qml.BitFlip(p, wires=wire)`) exactly when e0 == e1 == p.
    """
    k0 = np.array([[math.sqrt(1.0 - e0), 0.0], [0.0, math.sqrt(1.0 - e1)]])
    k1 = np.array([[0.0, 0.0], [math.sqrt(e0), 0.0]])
    k2 = np.array([[0.0, math.sqrt(e1)], [0.0, 0.0]])
    for wire in wires:
        qml.QubitChannel([k0, k1, k2], wires=wire)


def device_channel(gate_p: float, e0: float, e1: float, wires):
    """The two error channels this device model carries, in order: the 2Q
    gate error on the circuit's one CNOT, then asymmetric readout error.

    No other channel is applied. Coherent gate error, 1Q gate error and
    T1/T2 relaxation are all left out because none of them is a published
    figure for these devices, and the two channels here already dominate: at
    n=13, readout alone accounts for nearly all of the predicted deficit, and
    the 2Q gate error is a distant second, roughly 6x-12x smaller. See
    `game_numbers.GATE_2Q_ERROR` and `game_numbers.PUBLISHED_READOUT`.
    """
    two_qubit_gate_error(gate_p, wires)
    asymmetric_bit_flip(e0, e1, wires)


def with_readout_noise(gates, channel, wires=(0, 1)):
    """Wrap a bare gate function with a noise channel applied after it.

    `channel` is a callable of `wires` alone, already bound to its noise
    parameters (`functools.partial(device_channel, gate_p, e0, e1)`), so one
    call site works regardless of how many parameters the channel takes.
    """
    def circuit():
        gates()
        channel(wires)
    return circuit


def game_win_rate(dev, questions, circuits, channel) -> float:
    """Mean win rate of one sweep, each question's circuit run through `channel`."""
    win_rates = []
    for (x, y), gates in zip(questions, circuits):
        noisy = with_readout_noise(gates, channel)

        @qml.qnode(dev)
        def circuit():
            noisy()
            return qml.probs(wires=[0, 1])

        probs = qml.set_shots(circuit, shots=SHOTS)()
        win_rates.append(sum(
            probs[2 * a + b]
            for a in (0, 1)
            for b in (0, 1)
            if is_win(x, y, a, b)
        ))
    return sum(win_rates) / len(win_rates)


if __name__ == "__main__":
    n = N
    dev = noisy_device(wires=2)
    questions = question_order(n)
    circuits = []
    circuits.append([build_circuits(n, strategy_angle(n)), "precomputed"])
    circuits.append([build_mathmatical(n, strategy_angle(n)), "mathimatical"])

    ideal = omega_q(n)

    for circuit in circuits:
        print(circuit[1])
        print(f"omega_q(n={n}) = {ideal:.4f}  (noiseless quantum win rate)")
        print(f"2Q gate error (both devices): {GATE_2Q_ERROR:.4f}")
        print()
        print("device model: one 2Q-gate-error hit on the CNOT + asymmetric readout")
        rates = {}
        for qpu, (e0, e1, source) in PUBLISHED_READOUT.items():
            channel = partial(device_channel, GATE_2Q_ERROR, e0, e1)
            rate = game_win_rate(dev, questions, circuit[0], channel)
            rates[qpu] = rate
            deficit = ideal - rate
            print(
                f"{qpu:<10} e0={e0:.4f} e1={e1:.4f}  win rate={rate:.4f}  "
                f"deficit={deficit:.4f}  ({source})"
            )
        low, high = min(rates.values()), max(rates.values())
        print(
            f"predicted win-rate range across devices: [{low:.4f}, {high:.4f}] "
            f"(deficit range [{ideal - high:.4f}, {ideal - low:.4f}])"
        )
        print()


