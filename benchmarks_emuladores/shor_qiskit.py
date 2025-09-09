from qiskit import *
from qiskit.visualization import plot_histogram
from math import pi
import time
import psutil
import os

def _7mod15():
        
    circ = QuantumCircuit(8)
    circ.x(4)
    circ.cx(0,5)
    circ.cx(0,6)
    circ.cx(1,4)
    circ.cx(1,6)
    for i in range(4,8):
        circ.ccx(0,1,i)
    gate = circ.to_gate()
    gate.name = "7^x mod 15"
    return gate


def QFT(n):
    qft_circ = QuantumCircuit(n)
    for i in range(n-1, -1, -1):
        qft_circ.h(i)
        
        for j in range(i - 1, -1, -1): 
            qft_circ.cp(pi/(2 ** (i - j)), j, i)
 
        
    for i in range(n // 2):
        qft_circ.swap(i, n - i - 1)
    gate = qft_circ.to_gate()
    gate.name = "QFT" + str(n)
    return  gate

#Medición
start_time = time.time()

circ = QuantumCircuit(8,4)
circ.h(range(4))
circ.append(_7mod15(), range(8))
circ.measure(range(4,8),range(4))
circ.barrier(range(8))
circ.append(QFT(4), range(4))
circ.measure(range(4), range(4))
#circ.draw(output = 'mpl')

backend = Aer.get_backend("qasm_simulator")
job = execute(circ, backend, shots = 10000000)
result = job.result()
counts = result.get_counts()
#plot_histogram(counts)

# con esto sabemos que 4 es raiz cuadrada ya que el mcd de 4 8 12 es 4
# sabiendo esto podemos calcular los factores primos de la siguiente manera:

import math

primer_factor = math.gcd(4-1, 15)
segundo_factor = math.gcd(4+1, 15)
print("factores primos:", primer_factor, segundo_factor)

#Mediciones

end_time = time.time()
elapsed_time = end_time - start_time

#Medir uso de memoria y CPU
process = psutil.Process(os.getpid())
memory_used = process.memory_info().rss / (1024 * 1024)  # en MB
cpu_percent = process.cpu_percent(interval=1.0)  # % de CPU usado en 1s

print(f"Tiempo de ejecución: {elapsed_time:.4f} segundos")
print(f"Memoria usada: {memory_used:.2f} MB")
print(f"Uso de CPU: {cpu_percent:.2f}%")
