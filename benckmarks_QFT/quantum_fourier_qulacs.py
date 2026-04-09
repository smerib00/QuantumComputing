import time
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import H, CNOT, RZ

def qft_circuit(n):
    circuit = QuantumCircuit(n)
    for i in range(n):
        circuit.add_gate(H(i))
        for j in range(i+1, n):
            angle = 3.141592653589793 / (2 ** (j - i))
            circuit.add_gate(RZ(j, angle))
            circuit.add_gate(CNOT(j, i))
    return circuit

qubits_list = list(range(5, 35, 2))

for n in qubits_list:
    print(f"Ejecutando QFT con {n} qubits...")

    state = QuantumState(n)
    circuit = qft_circuit(n)

    start_time = time.time()
    circuit.update_quantum_state(state)
    end_time = time.time()

    print(f"Tiempo: {end_time - start_time:.4f}s")