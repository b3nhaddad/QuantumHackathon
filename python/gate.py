from load import *


@qml.qnode(dev)
def bell_state(x):
    qml.H(0)
    #[0,1]
    qml.CNOT(wires=x)
    #now we are in a bell state
    #first bit is 00 or 11, allows us to know this other bit

