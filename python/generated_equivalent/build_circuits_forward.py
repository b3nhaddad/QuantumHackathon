def build_circuits(
    n: int,
    theta: float,
) -> list[Callable[[], None]]:

    # Exact-equivalent odd-cycle strategy selected by the local optimizer.
    #
    # gauge      = 0.0000000000000000
    # chirality  = +1
    # CNOT       = forward
    #
    # Gauge and chirality do not change the noiseless game value.
    gauge = 0.0000000000000000
    chirality = +1

    step = (
        chirality
        * (
            math.pi
            - 4.0 * theta
        )
    )

    bob_offset = (
        chirality
        * 2.0
        * theta
    )

    def gates_for(
        x: int,
        y: int,
    ) -> Callable[[], None]:

        angle_a = (
            gauge
            + x * step
        )

        angle_b = (
            gauge
            + y * step
            + bob_offset
        )

        def circuit() -> None:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])

            qml.RY(
                angle_a,
                wires=0,
            )

            qml.RY(
                angle_b,
                wires=1,
            )

        return circuit

    return [
        gates_for(x, y)
        for x, y
        in question_order(n)
    ]
