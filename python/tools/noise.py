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

from game_numbers import PUBLISHED_READOUT
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


def bit_flip(p: float, wires):
    """Symmetric readout error: each wire flips with probability p either way."""
    for wire in wires:
        qml.BitFlip(p, wires=wire)


def depolarizing(p: float, wires):
    for wire in wires:
        qml.DepolarizingChannel(p, wires=wire)


def amplitude_damping(gamma: float, wires):
    for wire in wires:
        qml.AmplitudeDamping(gamma, wires=wire)


def asymmetric_bit_flip(e0: float, e1: float, wires):
    """Readout error that reads a true 0 as 1 w.p. e0 and a true 1 as 0 w.p. e1.

    Kraus form of the channel `game_numbers.delta_from_readout_asym` derives
    its formulas for: K0 leaves each basis state alone with its own survival
    amplitude, K1 carries |0> to |1> at rate e0, K2 carries |1> to |0> at rate
    e1. Reduces to `bit_flip(p, wires)` exactly when e0 == e1 == p.
    """
    k0 = np.array([[math.sqrt(1.0 - e0), 0.0], [0.0, math.sqrt(1.0 - e1)]])
    k1 = np.array([[0.0, 0.0], [math.sqrt(e0), 0.0]])
    k2 = np.array([[0.0, math.sqrt(e1)], [0.0, 0.0]])
    for wire in wires:
        qml.QubitChannel([k0, k1, k2], wires=wire)


def with_readout_noise(gates, channel, wires=(0, 1)):
    """Wrap a bare gate function with a noise channel applied after it.

    `channel` is a callable of `wires` alone, already bound to its noise
    parameters (`functools.partial(bit_flip, p)`,
    `functools.partial(asymmetric_bit_flip, e0, e1)`, ...), so one call site
    works for every channel regardless of how many parameters it takes.
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
    circuits.append([build_circuits(n, strategy_angle(n)),"precomputed"])
    circuits.append([ build_mathmatical(n, strategy_angle(n)),"mathimatical"])

    symmetric_channels = {
        "bit_flip": bit_flip,
        "depolarizing": depolarizing,
        "amplitude_damping": amplitude_damping,
    }

    for circuit in circuits:
        for name, channel_fn in symmetric_channels.items():
            for p in (0.0, 0.01, 0.05, 0.1):
                rate = game_win_rate(dev, questions, circuit[0], partial(channel_fn, p))
                print(f"{name:<18} p={p:.2f}  win rate={rate:.4f}")
        print(circuit[1])
        print()
        print("asymmetric readout, PUBLISHED_READOUT (e0, e1):")
        for qpu, (e0, e1, source) in PUBLISHED_READOUT.items():
            rate = game_win_rate(dev, questions, circuit[0], partial(asymmetric_bit_flip, e0, e1))
            print(f"{qpu:<10} e0={e0:.4f} e1={e1:.4f}  win rate={rate:.4f}  ({source})")


