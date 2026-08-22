import pennylane as qml

def get_device(wires: int, shots=None):
    devices = [
        "lightning.gpu",    # NVIDIA GPU
        "lightning.qubit",  # Optimized CPU
        "default.qubit",    # Pure PennyLane fallback
    ]

    for name in devices:
        try:
            dev = qml.device(
                name,
                wires=wires,
                shots=shots,
            )

            # Force device initialization to catch GPU problems
            @qml.qnode(dev)
            def test_circuit():
                qml.Hadamard(0)
                return qml.expval(qml.PauliZ(0))

            test_circuit()

            print(f"Using device: {name}")
            return dev

        except Exception as e:
            print(f"{name} unavailable: {e}")

    raise RuntimeError("No usable PennyLane device found.")

# 1. Define a device with 2 qubit (wire)

dev = get_device(wires=2)

