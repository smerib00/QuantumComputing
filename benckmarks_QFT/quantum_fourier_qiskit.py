import time
import tracemalloc
import csv
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from qiskit import transpile
import psutil
import os

# Configuración
qubits_list = list(range(21, 36, 2))  # de 5 a 25 qubits
shots = 1  # simulación de estado, no sampling

simulator = AerSimulator(method="statevector")

results = []

for n in qubits_list:
    print(f"Ejecutando QFT con {n} qubits...")

    # Crear circuito QFT
    qc = QuantumCircuit(n)
    qc.append(QFT(n), range(n))
    qc.measure_all()

    # Medir memoria
    #tracemalloc.start()
    #process = psutil.Process(os.getpid())

    #mem_before = process.memory_info().rss

    start_time = time.time()

    # Ejecutar simulación
    qc = transpile(qc, simulator)
    job = simulator.run(qc)

    result = job.result()

    end_time = time.time()

    #current, peak = tracemalloc.get_traced_memory()
    #tracemalloc.stop()

    execution_time = end_time - start_time
    #peak_memory_mb = peak / 10**6

    #mem_after = process.memory_info().rss

    #memory_mb = (mem_after - mem_before) / 10**6
    print(f"Tiempo: {execution_time:.4f}s")
#    print(f"Tiempo: {execution_time:.4f}s | Memoria pico: {peak_memory_mb:.2f} MB")

    results.append({
        "qubits": n,
        "time_sec": execution_time,
      #  "memory_mb": memory_mb
    })

# Guardar resultados
hostname = socket.gethostname()
filename = f"qft_results_{hostname}.csv"
with open(filename, "w", newline="") as csvfile:
    fieldnames = ["qubits", "time_sec"]#, "memory_mb"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(results)

print("Resultados guardados en {filename}")
