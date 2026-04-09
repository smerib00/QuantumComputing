import time
import csv
import pennylane as qml
from pennylane import numpy as np

qubits_list = list(range(31, 33, 1))

results = []

for n in qubits_list:
    print(f"Ejecutando QFT con {n} qubits...")

    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit():
        qml.QFT(wires=range(n))
        return qml.state()

    start_time = time.time()
    circuit()
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Tiempo: {execution_time:.4f}s")

    results.append({"qubits": n, "time_sec": execution_time})
