import sys
from pathlib import Path

import pennylane as qml

# solution.py is a sibling of this file's parent (python/), not of tools/,
# so it isn't importable by name until that directory is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solution import build_circuits, is_win, question_order, strategy_angle

# Noise needs density matrices, not statevectors, so this device is fixed
# regardless of what load.get_device picked for the noiseless run.
def noisy_device(wires: int, shots=None):
    return qml.device("default.mixed", wires=wires, shots=shots)


def bit_flip(p: float, wires):
    for wire in wires:
        qml.BitFlip(p, wires=wire)


def depolarizing(p: float, wires):
    for wire in wires:
        qml.DepolarizingChannel(p, wires=wire)


def amplitude_damping(gamma: float, wires):
    for wire in wires:
        qml.AmplitudeDamping(gamma, wires=wire)


def with_readout_noise(gates, channel=bit_flip, p: float = 0.0, wires=(0, 1)):
    """Wrap a bare gate function with a per-wire noise channel applied after it."""
    def circuit():
        gates()
        channel(p, wires)
    return circuit


if __name__ == "__main__":
    n = 3
    dev = noisy_device(wires=2)
    questions = question_order(n)
    circuits = build_circuits(n, strategy_angle(n))

    for p in (0.0, 0.01, 0.05, 0.1):
        win_rates = []
        for (x, y), gates in zip(questions, circuits):
            noisy = with_readout_noise(gates, bit_flip, p)

            @qml.qnode(dev)
            def circuit():
                noisy()
                return qml.probs(wires=[0, 1])

            probs = circuit()
            win_rates.append(sum(
                probs[2 * a + b]
                for a in (0, 1)
                for b in (0, 1)
                if is_win(x, y, a, b)
            ))

        print(f"p={p:.2f}  win rate={sum(win_rates) / len(win_rates):.4f}")

