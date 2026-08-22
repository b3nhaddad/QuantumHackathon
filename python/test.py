import pennylane as qml
import pytest

from load import get_device

# Unit testing boilerplate

def test_get_device_returns_device():
    dev = get_device(wires=2)

    assert dev is not None


def test_device_has_correct_number_of_wires():
    dev = get_device(wires=4)

    assert len(dev.wires) == 4


def test_simple_circuit_runs():
    dev = get_device(wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(0)
        qml.CNOT(wires=[0, 1])

        return qml.probs(wires=[0, 1])

    result = circuit()

    assert len(result) == 4
    assert result.sum() == pytest.approx(1.0)


def test_bell_state_probabilities():
    dev = get_device(wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(0)
        qml.CNOT(wires=[0, 1])

        return qml.probs(wires=[0, 1])

    result = circuit()

    assert result[0] == pytest.approx(0.5)
    assert result[1] == pytest.approx(0.0)
    assert result[2] == pytest.approx(0.0)
    assert result[3] == pytest.approx(0.5)

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
