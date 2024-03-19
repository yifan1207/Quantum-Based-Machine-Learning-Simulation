import numpy as np 
import qutip as qt
import cirq as cq 
N = 100 # Number of qubit
M = 1000 # Number of compounds
T = 10 # Number of iterations
criteria = 0.8 # Threshold for filtering compounds

# Generating random compounds as qubit states
compounds = [qt.rand_ket(N) for _ in range(M)]

# Defining the target protein as a qubit state
target = qt.rand_ket(N)

# Defining the quantum simulation device as a cirq circuit
device = cq.Circuit()

# Adding gates to the device to simulate the interaction between compounds and target
for i in range(N):
    device.append(cq.CNOT(target[i], compounds[0][i])) # Entangle the target with the first compounds
    for j in range(1, M):
        device.append(cq.SWAP(compounds[j-1][i], compounds[j][i])) # Swap the compounds along the chain
    device.append(cq.CNOT(target[i], compounds[M-1][i])) # Entangle the target with the last compound

# Running the device for T iterations
for t in range(T):
    device.append(device) # Repeat the circuit

# Measuring the final states of the compounds
results = device.final_state_vector()

# Filtering the compounds based on the criteria
filtered = []
for i in range(M):
    # Calculate the fidelity between the compound and the target
    fidelity = qt.fidelity(results[i], target)
    # If the fidelity is above the criteria, add the compound to the filtered list
    if fidelity > criteria:
        filtered.append(results[i])

# Printing the number of filtered compounds
print(f"Number of filtered compounds: {len(filtered)}")

# prototype
