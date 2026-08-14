import pennylane as qml
from pennylane import numpy as np

# Number of qubits
n_qubits = 3

# Quantum device
dev = qml.device("default.qubit", wires=n_qubits)


# Oracle for target state |101>
def oracle():
    qml.PauliX(wires=1)
    qml.ctrl(qml.PauliZ, control=[0, 1])(wires=2)
    qml.PauliX(wires=1)


# Diffusion operator
def diffusion():

    for i in range(n_qubits):
        qml.Hadamard(wires=i)

    for i in range(n_qubits):
        qml.PauliX(wires=i)

    qml.Hadamard(wires=2)
    qml.ctrl(qml.PauliX, control=[0, 1])(wires=2)
    qml.Hadamard(wires=2)

    for i in range(n_qubits):
        qml.PauliX(wires=i)

    for i in range(n_qubits):
        qml.Hadamard(wires=i)


iterations = 2


@qml.qnode(dev)
def grover():

    # Initialize equal superposition
    for i in range(n_qubits):
        qml.Hadamard(wires=i)

    # Apply Grover iterations
    for _ in range(iterations):
        oracle()
        diffusion()

    return qml.probs(wires=range(n_qubits))


# Run the algorithm
probabilities = grover()

states = [format(i, "03b") for i in range(2 ** n_qubits)]

print("Grover Search Results:\n")

for state, prob in zip(states, probabilities):
    print(f"{state} : {prob:.4f}")

print("\nMost probable state:", states[np.argmax(probabilities)])