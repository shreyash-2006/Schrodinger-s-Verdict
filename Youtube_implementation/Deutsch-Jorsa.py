import pennylane as qml

n = 3

dev = qml.device("default.qubit", wires=n+1)


# ------------------------
# Oracles
# ------------------------

def constant_oracle():
    pass


def balanced_oracle():
    qml.CNOT(wires=[0,3])


@qml.qnode(dev)
def deutsch_jozsa(oracle):

    # Prepare ancilla
    qml.PauliX(wires=3)

    # Put all qubits into superposition
    for i in range(n+1):
        qml.Hadamard(wires=i)

    # Oracle
    oracle()

    # Hadamards again
    for i in range(n):
        qml.Hadamard(wires=i)

    return qml.probs(wires=range(n))


print("Constant Oracle")
print(deutsch_jozsa(constant_oracle))

print()

print("Balanced Oracle")
print(deutsch_jozsa(balanced_oracle))
drawer = qml.draw(deutsch_jozsa)

print(drawer(balanced_oracle))