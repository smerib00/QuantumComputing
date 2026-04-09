import time
from qibo import Circuit, gates

def qft(n):
    circuit = Circuit(n)
    for i in range(n):
        circuit.add(gates.H(i))
        for j in range(i+1, n):
            theta = 3.141592653589793 / (2 ** (j - i))
            circuit.add(gates.CU1(j, i, theta))
    return circuit

qubits_list = list(range(31, 32, 1))

for n in qubits_list:
    print(f"Ejecutando QFT con {n} qubits...")

    circuit = qft(n)

    start_time = time.time()
    result = circuit()
    end_time = time.time()

    print(f"Tiempo: {end_time - start_time:.4f}s")
