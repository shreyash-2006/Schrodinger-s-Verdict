import sys
print("Python Executable being used:", sys.executable)
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch import nn

data = load_breast_cancer()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target

X = df.drop(['target'], axis=1)
y = df['target']

algorithm_globals.random_seed = 14
np.random.seed(algorithm_globals.random_seed)
torch.manual_seed(algorithm_globals.random_seed)

algorithm_globals.random_seed = 14
np.random.seed(algorithm_globals.random_seed)
torch.manual_seed(algorithm_globals.random_seed)

X_train_raw, X_test_raw, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=algorithm_globals.random_seed, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

pca = PCA(n_components=4)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Cumulative variance explained:", pca.explained_variance_ratio_.sum())

mm_scaler = MinMaxScaler(feature_range=(0, 2*np.pi))
train_features = mm_scaler.fit_transform(X_train_pca)
test_features = mm_scaler.transform(X_test_pca)


num_features = train_features.shape[1]

feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
ansatz = RealAmplitudes(num_qubits=num_features, reps=4)

observables = []
for i in range(num_features):
    pauli_str = "I" * i + "Z" + "I" * (num_features - i - 1)
    observables.append(SparsePauliOp(pauli_str))

estimator_qnn = EstimatorQNN(
    circuit=feature_map.compose(ansatz),
    observables=observables,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters
)

initial_weights = algorithm_globals.random.random(estimator_qnn.num_weights)

quantum_layer = TorchConnector(estimator_qnn, initial_weights=initial_weights)

model = nn.Sequential(
    quantum_layer,
    nn.Linear(4, 2)
)

X_train_tensor = torch.tensor(train_features, dtype=torch.float32)
y_train_tensor = torch.tensor(train_labels.to_numpy(), dtype=torch.long)

X_test_tensor = torch.tensor(test_features, dtype=torch.float32)
y_test_tensor = torch.tensor(test_labels.to_numpy(), dtype=torch.long)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

epochs = 100
loss_history = []

for epoch in range(epochs):
    epoch_losses = []
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
    loss_history.append(avg_epoch_loss)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.4f}")



model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    predicted_labels = torch.argmax(test_outputs, dim=1)
    test_accuracy = (predicted_labels == y_test_tensor).float().mean().item()

print("Test accuracy:", test_accuracy)