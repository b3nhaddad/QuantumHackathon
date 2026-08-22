import pennylane as qml

# 1. Define a device with 1 qubit (wire)
dev = qml.device("default.qubit", wires=1)

# 2. Define the quantum circuit using a QNode
@qml.qnode(dev)
def circuit(theta):
    qml.RY(theta, wires=0)
    return qml.expval(qml.PauliZ(0))
