import pennylane as qml

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
    dev = noisy_device(wires=2)

    def bell_pair():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])

    for p in (0.0, 0.01, 0.05, 0.1):
        noisy = with_readout_noise(bell_pair, bit_flip, p)

        @qml.qnode(dev)
        def circuit():
            noisy()
            return qml.probs(wires=[0, 1])

        print(f"p={p:.2f}  probs={circuit()}")
