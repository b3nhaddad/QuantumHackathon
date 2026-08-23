TEAM = "your_team_name_10"
RUN = 1
N = 21
SHOTS = 110
TWIRLS = 1
QPU = "emerald"
DELTA = 0.015
DEVICE = "default.qubit"

# ==========================================================================
# THE PART YOUR TEAM WRITES
# ==========================================================================


def build_circuits(
    n: int,
    theta: float,
) -> list[Callable[[], None]]:

    step = math.pi - 4.0 * theta
    bob_offset = 2.0 * theta

    def gates_for(
        x: int,
        y: int,
    ) -> Callable[[], None]:

        angle_a = x * step
        angle_b = y * step + bob_offset

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


